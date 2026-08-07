import socket
import struct
import threading
import unittest
from unittest.mock import patch

from vconsole2_mcp import (
    NetConsoleClient,
    NetConsoleTarget,
    RconClient,
    RconTarget,
    SERVERDATA_AUTH,
    SERVERDATA_AUTH_RESPONSE,
    SERVERDATA_EXECCOMMAND,
    SERVERDATA_RESPONSE_VALUE,
    RconError,
    arbitrary_local_commands_enabled,
    handle_tool,
    safe_filename,
    safe_run_name,
    validate_addon_name,
    validate_local_command,
)
from local_lab_runner import controlled_hero_ready, passive_opponent_ready, stop_owned_bridge


def send_packet(connection, request_id, packet_type, body):
    payload = struct.pack("<ii", request_id, packet_type) + body.encode() + b"\0\0"
    connection.sendall(struct.pack("<i", len(payload)) + payload)


def read_packet(connection):
    size = struct.unpack("<i", connection.recv(4))[0]
    payload = connection.recv(size)
    return (*struct.unpack("<ii", payload[:8]), payload[8:-2].decode())


class FakeRconServer:
    def __enter__(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.listener.close()
        self.thread.join(timeout=1)

    def run(self):
        connection, _ = self.listener.accept()
        with connection:
            request_id, packet_type, password = read_packet(connection)
            assert packet_type == SERVERDATA_AUTH and password == "test-password"
            send_packet(connection, request_id, SERVERDATA_AUTH_RESPONSE, "")
            request_id, packet_type, command = read_packet(connection)
            assert packet_type == SERVERDATA_EXECCOMMAND
            send_packet(connection, request_id, SERVERDATA_RESPONSE_VALUE, command[:3])
            send_packet(connection, request_id, SERVERDATA_RESPONSE_VALUE, command[3:])


class RconClientTests(unittest.TestCase):
    def test_authenticates_and_collects_split_command_output(self):
        with FakeRconServer() as server:
            target = RconTarget("127.0.0.1", server.port, "test-password")
            with RconClient(target, timeout=1) as client:
                self.assertEqual(client.execute("status", timeout=1), "status")


class NetConsoleClientTests(unittest.TestCase):
    def test_sends_a_newline_delimited_command_and_returns_output(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve():
            connection, _ = listener.accept()
            with connection:
                self.assertEqual(connection.recv(64), b"status\n")
                connection.sendall(b"hostname: test\n")

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            with NetConsoleClient(NetConsoleTarget("127.0.0.1", port), timeout=1) as client:
                self.assertEqual(client.execute("status", timeout=1), "hostname: test\n")
        finally:
            listener.close()
            thread.join(timeout=1)

    def test_reads_bounded_logs_without_sending_a_command(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve():
            connection, _ = listener.accept()
            with connection:
                connection.sendall(b"first\nsecond\nthird\n")

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            with NetConsoleClient(NetConsoleTarget("127.0.0.1", port), timeout=1) as client:
                output = client.read_logs(max_lines=2, max_bytes=1024, timeout=1)
            self.assertEqual(output, "first\nsecond\n[truncated by local MCP limit]")
        finally:
            listener.close()
            thread.join(timeout=1)


class LocalBoundaryTests(unittest.TestCase):
    def test_default_command_allowlist_and_custom_override(self):
        validate_local_command("status")
        with self.assertRaises(RconError):
            validate_local_command("quit")
        with patch.dict("os.environ", {"VCONSOLE_LOCAL_COMMANDS": "status, rl_ppo_progress"}, clear=False):
            validate_local_command("rl_ppo_progress")
            with self.assertRaises(RconError):
                validate_local_command("rl_ppo_restart")

    def test_bridge_inputs_must_be_filenames(self):
        self.assertEqual(safe_filename("candidate.pt", suffix=".pt", label="checkpoint_name"), "candidate.pt")
        with self.assertRaises(RconError):
            safe_filename("..\\candidate.pt", suffix=".pt", label="checkpoint_name")

    def test_arbitrary_commands_require_explicit_environment_opt_in(self):
        self.assertFalse(arbitrary_local_commands_enabled())
        with patch.dict("os.environ", {"VCONSOLE_ALLOW_ARBITRARY_LOCAL_COMMANDS": "true"}, clear=False):
            self.assertTrue(arbitrary_local_commands_enabled())

    def test_addon_and_map_names_reject_console_injection_characters(self):
        self.assertEqual(validate_addon_name("rl_ppo_local", "addon"), "rl_ppo_local")
        with self.assertRaises(RconError):
            validate_addon_name("rl_ppo_local; quit", "addon")

    def test_lab_defaults_launch_the_template_map(self):
        with patch("vconsole2_mcp.launch_local_addon", return_value="") as launch:
            result = handle_tool("local_addon_launch", {})
        self.assertNotIn("isError", result)
        self.assertEqual(launch.call_args.args[:2], ("rl_ppo_local", "template_map"))

    def test_run_name_cannot_be_a_path(self):
        self.assertEqual(safe_run_name("lane_v3_candidate"), "lane_v3_candidate")
        with self.assertRaises(RconError):
            safe_run_name("../overwrite")

    def test_runner_joins_local_team_before_accepting_hero(self):
        with patch("local_lab_runner.execute_local_console", side_effect=["", "RL PPO progression: level=1 xp=0"] ) as execute:
            self.assertTrue(controlled_hero_ready())
        self.assertEqual(execute.call_args_list[0].args[0], "jointeam good")
        self.assertEqual(execute.call_args_list[1].args[0], "rl_ppo_progress")

    def test_runner_requires_the_passive_shadow_fiend_before_collection(self):
        with patch("local_lab_runner.execute_local_console", return_value=(
                "RL PPO opponent: mode=passive identity=passive_nevermore_v1 team=3")) as execute:
            self.assertTrue(passive_opponent_ready())
        self.assertEqual(execute.call_args.args[0], "rl_ppo_opponent")
        with patch("local_lab_runner.execute_local_console", return_value="RL PPO opponent: unavailable"):
            self.assertFalse(passive_opponent_ready())

    def test_runner_stops_only_its_recorded_bridge_process_tree(self):
        with patch("local_lab_runner.subprocess.run") as terminate:
            stop_owned_bridge(1234)
            stop_owned_bridge(0)
        terminate.assert_called_once()
        self.assertEqual(terminate.call_args.args[0], ["taskkill", "/PID", "1234", "/T", "/F"])


if __name__ == "__main__":
    unittest.main()
