-- Reference adapter for a LOCAL Dota custom lobby. It assumes a JSON Lua
-- library is available as `json`, and a local bridge on 127.0.0.1:8765.
-- Do not put this into a public matchmaking bot script.

require("game/dkjson")

if RLPPOBridge == nil then RLPPOBridge = class({}) end

local ACTION_COUNT = 24
local OBSERVATION_DIM = 32
local HEALTH_BAR_SEGMENTS = 20
local DIRECTIONS = {
    move_north = Vector(0, 1, 0), move_north_east = Vector(1, 1, 0),
    move_east = Vector(1, 0, 0), move_south_east = Vector(1, -1, 0),
    move_south = Vector(0, -1, 0), move_south_west = Vector(-1, -1, 0),
    move_west = Vector(-1, 0, 0), move_north_west = Vector(-1, 1, 0),
}
local DIRECTION_ACTIONS = {"move_north", "move_north_east", "move_east", "move_south_east",
    "move_south", "move_south_west", "move_west", "move_north_west"}

local function clamp(value, low, high)
    return math.max(low, math.min(value, high))
end

local function health_bar_fraction(health, max_health)
    local fraction = clamp(health / math.max(max_health, 1), 0, 1)
    return math.floor(fraction * HEALTH_BAR_SEGMENTS + 0.5) / HEALTH_BAR_SEGMENTS
end

function RLPPOBridge:Start(hero, player_id, base_url)
    self.hero = hero
    self.player_id = player_id
    self.base_url = base_url or "http://127.0.0.1:8765"
    self.pending = nil
    self.previous_origin = hero:GetAbsOrigin()
    self.previous_time = GameRules:GetGameTime()
    self.previous_score = 0
    self.episode_last_hits = 0
    self.calibration_last_hit = false
    self.calibration_hero_dead = false
    self.decision_interval = 0.25
    self.enabled = true
    self.request_in_flight = false
    self.finish_requested = false
    self.rotation_requested = false
    self.episode_end_requested = false
    self.bonus_reward = 0
    self.previous_creep_distance = 800
    self.previous_target_entindex = -1
    self.previous_target_health_bar = 0
    self.previous_target_time = self.previous_time
    self.reward_version = "sf1v1_passive_v1"
    self.opponent = nil
    self.opponent_mode = "absent"
    self.opponent_identity = "none"
end

function RLPPOBridge:SetOpponent(opponent, mode, identity)
    self.opponent = opponent
    self.opponent_mode = mode or "absent"
    self.opponent_identity = identity or "none"
end

function RLPPOBridge:EnvironmentMetadata()
    local opponent = self.opponent
    local opponent_team = nil
    local opponent_name = nil
    if opponent ~= nil and not opponent:IsNull() then
        opponent_team = opponent:GetTeamNumber()
        opponent_name = opponent:GetUnitName()
    end
    return {contract = "sf1v1_passive_v1", map = "template_map",
        controlled_hero = self.hero:GetUnitName(), controlled_team = self.hero:GetTeamNumber(),
        opponent_mode = self.opponent_mode, opponent_identity = self.opponent_identity,
        opponent_hero = opponent_name, opponent_team = opponent_team}
end

function RLPPOBridge:Post(path, payload, callback)
    local request = CreateHTTPRequestScriptVM("POST", self.base_url .. path)
    request:SetHTTPRequestHeaderValue("Content-Type", "application/json")
    request:SetHTTPRequestRawPostBody("application/json", json.encode(payload))
    request:SetHTTPRequestAbsoluteTimeoutMS(1000)
    request:Send(function(result)
        if result.StatusCode == 200 then
            local response, _, error_text = json.decode(result.Body)
            if response == nil then print("RL PPO bridge returned invalid JSON: " .. tostring(error_text)) end
            callback(response)
        else
            print("RL PPO bridge request failed: " .. tostring(result.StatusCode) .. " " .. tostring(result.Body))
            callback(nil)
        end
    end)
end

function RLPPOBridge:Observation()
    local origin = self.hero:GetAbsOrigin()
    local now = GameRules:GetGameTime()
    local elapsed = math.max(now - self.previous_time, 0.001)
    local velocity = (origin - self.previous_origin) / elapsed
    self.previous_origin = origin
    self.previous_time = now
    local gold = PlayerResource:GetGold(self.player_id)
    local last_hits = self.episode_last_hits
    local xp = self.hero:GetCurrentXP()
    local distance = math.sqrt(origin.x * origin.x + origin.y * origin.y)
    local target, enemy_count, ally_count, nearest_ally, allies = self:LaneCreepContext()
    local target_distance = 800
    local attack_range = math.max(self.hero:Script_GetAttackRange(), 1)
    local damage = 1
    local target_health, target_max_health, target_entindex = 0, 1, -1
    local creep_dx, creep_dy, creep_distance, creep_health_bar = 0, 0, 0, 0
    local health_loss_rate, hero_damage_fraction, in_attack_range, allied_pressure = 0, 0, 0, 0
    local ally_dx, ally_dy, ally_distance, ally_health_bar = 0, 0, 0, 0
    local attack_recovery = 0
    local opponent_present, opponent_dx, opponent_dy, opponent_distance = 0, 0, 0, 0
    local opponent_health_bar, opponent_mana_bar, opponent_facing_to_hero = 0, 0, 0
    -- This value is observed from the live hero whenever the binding supports
    -- it. It is also exported to calibration telemetry so the offline
    -- simulator does not need to keep guessing a fixed attack period.
    local seconds_per_attack = 0.75
    if target ~= nil then
        local delta = target:GetAbsOrigin() - origin
        target_distance = delta:Length2D()
        damage = math.max(self.hero:GetAverageTrueAttackDamage(target), 1)
        target_health = target:GetHealth()
        target_max_health = math.max(target:GetMaxHealth(), 1)
        target_entindex = target:entindex()
        creep_dx = clamp(delta.x / 800, -1, 1)
        creep_dy = clamp(delta.y / 800, -1, 1)
        creep_distance = math.min(target_distance / 800, 1.5)
        creep_health_bar = health_bar_fraction(target_health, target_max_health)
        if self.previous_target_entindex == target_entindex then
            health_loss_rate = clamp((self.previous_target_health_bar - creep_health_bar) /
                math.max(now - self.previous_target_time, 0.001), -1, 1)
        end
        self.previous_target_entindex = target_entindex
        self.previous_target_health_bar = creep_health_bar
        self.previous_target_time = now
        hero_damage_fraction = clamp(damage / target_max_health, 0, 1)
        in_attack_range = target_distance <= attack_range and 1 or 0
        for _, ally in pairs(allies) do
            if ally.GetAttackTarget ~= nil and ally:GetAttackTarget() == target then
                allied_pressure = allied_pressure + 1
            end
        end
        allied_pressure = math.min(allied_pressure / 4, 1)
    else
        self.previous_target_entindex = -1
        self.previous_target_health_bar = 0
        self.previous_target_time = now
    end
    if nearest_ally ~= nil then
        local ally_delta = nearest_ally:GetAbsOrigin() - origin
        ally_dx = clamp(ally_delta.x / 800, -1, 1)
        ally_dy = clamp(ally_delta.y / 800, -1, 1)
        ally_distance = math.min(ally_delta:Length2D() / 800, 1.5)
        ally_health_bar = health_bar_fraction(nearest_ally:GetHealth(), nearest_ally:GetMaxHealth())
    end
    if self.hero.GetLastAttackTime ~= nil and self.hero.GetSecondsPerAttack ~= nil then
        -- Current Dota builds bind GetSecondsPerAttack with one explicit
        -- parameter.  Keep the calibrated fallback if an older/newer binding
        -- rejects it, so an observation feature can never crash a rollout.
        local ok, observed_seconds = pcall(function()
            return self.hero:GetSecondsPerAttack(false)
        end)
        if ok and type(observed_seconds) == "number" and observed_seconds > 0 then
            seconds_per_attack = observed_seconds
        end
        attack_recovery = clamp((self.hero:GetLastAttackTime() + seconds_per_attack - now) / seconds_per_attack, 0, 1)
    end
    local opponent = self.opponent
    if opponent ~= nil and not opponent:IsNull() and opponent:IsAlive() then
        local opponent_origin = opponent:GetAbsOrigin()
        local opponent_delta = opponent_origin - origin
        local opponent_length = opponent_delta:Length2D()
        opponent_present = 1
        opponent_dx = clamp(opponent_delta.x / 800, -1, 1)
        opponent_dy = clamp(opponent_delta.y / 800, -1, 1)
        opponent_distance = math.min(opponent_length / 800, 1.5)
        opponent_health_bar = health_bar_fraction(opponent:GetHealth(), opponent:GetMaxHealth())
        opponent_mana_bar = health_bar_fraction(opponent:GetMana(), math.max(opponent:GetMaxMana(), 1))
        if opponent_length > 0 and opponent.GetForwardVector ~= nil then
            local toward_hero = (origin - opponent_origin):Normalized()
            opponent_facing_to_hero = clamp(opponent:GetForwardVector():Dot(toward_hero), -1, 1)
        end
    end
    local observation = {origin.x / 8192, origin.y / 8192, velocity.x / 550, velocity.y / 550,
        math.max(now, 0) / 3600, gold / 30000, last_hits / 400, xp / 30000, 1, distance / 11585,
        creep_dx, creep_dy, creep_distance, creep_health_bar, health_loss_rate, hero_damage_fraction,
        in_attack_range, allied_pressure, ally_dx, ally_dy, ally_distance, ally_health_bar,
        math.min(enemy_count / 4, 1.5), math.min(ally_count / 4, 1.5), attack_recovery,
        opponent_present, opponent_dx, opponent_dy, opponent_distance, opponent_health_bar,
        opponent_mana_bar, opponent_facing_to_hero}
    if #observation ~= OBSERVATION_DIM then error("RL PPO observation dimension mismatch") end
    local mask = {}
    for index = 1, ACTION_COUNT do mask[index] = false end
    mask[1] = true -- idle
    if target ~= nil then
        local delta = target:GetAbsOrigin() - origin
        -- Curriculum guardrail: explore toward the wave, not away from it.
        for index, action_name in ipairs(DIRECTION_ACTIONS) do
            local direction = DIRECTIONS[action_name]
            mask[index + 1] = direction.x * delta.x + direction.y * delta.y >= 0
        end
        mask[10] = true -- action index 9 = attack
    else
        for index = 2, 9 do mask[index] = true end
    end
    local calibration = {schema_version = 1, event = "transition", game_time = now,
        target_distance = target_distance, attack_range = attack_range, attack_damage = damage,
        hero_move_speed = self.hero:GetIdealSpeed(), target_health = target_health,
        target_max_health = target_max_health, target_entindex = target_entindex,
        hero_health = self.hero:GetHealth(), hero_max_health = math.max(self.hero:GetMaxHealth(), 1),
        enemy_count = enemy_count, ally_count = ally_count,
        hero_attack_cooldown = seconds_per_attack, attack_recovery = attack_recovery,
        in_attack_range = in_attack_range}
    return observation, mask, calibration
end

function RLPPOBridge:LaneCreepContext()
    local origin = self.hero:GetAbsOrigin()
    local enemy = FindUnitsInRadius(self.hero:GetTeamNumber(), origin, nil, 800,
        DOTA_UNIT_TARGET_TEAM_ENEMY, DOTA_UNIT_TARGET_BASIC, DOTA_UNIT_TARGET_FLAG_NONE, FIND_CLOSEST, false)
    local ally = FindUnitsInRadius(self.hero:GetTeamNumber(), origin, nil, 800,
        DOTA_UNIT_TARGET_TEAM_FRIENDLY, DOTA_UNIT_TARGET_BASIC, DOTA_UNIT_TARGET_FLAG_NONE, FIND_CLOSEST, false)
    return enemy[1], #enemy, #ally, ally[1], ally
end

function RLPPOBridge:Reward()
    local score = PlayerResource:GetGold(self.player_id) + 0.05 * self.hero:GetCurrentXP()
    local target = self:NearestEnemy()
    local creep_distance = 800
    if target ~= nil then creep_distance = math.min((target:GetAbsOrigin() - self.hero:GetAbsOrigin()):Length2D(), 800) end
    local approach_reward = math.max(self.previous_creep_distance - creep_distance, 0) / 800 * 0.02
    local reward = (score - self.previous_score) / 1000 + approach_reward + self.bonus_reward
    self.previous_score = score
    self.previous_creep_distance = creep_distance
    self.bonus_reward = 0
    return reward
end

function RLPPOBridge:AddReward(value, reason)
    self.bonus_reward = self.bonus_reward + value
    if reason == "last_hit" then
        self.calibration_last_hit = true
        self.episode_last_hits = self.episode_last_hits + 1
    end
    if reason == "hero_death" then self.calibration_hero_dead = true end
end

function RLPPOBridge:NearestEnemy()
    local target = self:LaneCreepContext()
    return target
end

function RLPPOBridge:Execute(action_name)
    if action_name == "idle" then return true end
    if DIRECTIONS[action_name] ~= nil then
        local direction = DIRECTIONS[action_name]:Normalized()
        self:IssuePolicyOrder({UnitIndex = self.hero:entindex(), OrderType = DOTA_UNIT_ORDER_MOVE_TO_POSITION,
            Position = self.hero:GetAbsOrigin() + direction * 300, Queue = 0})
        return true
    end
    if action_name == "stop" then
        self:IssuePolicyOrder({UnitIndex = self.hero:entindex(), OrderType = DOTA_UNIT_ORDER_STOP, Queue = 0})
        return true
    end
    if action_name == "hold" then
        self:IssuePolicyOrder({UnitIndex = self.hero:entindex(), OrderType = DOTA_UNIT_ORDER_HOLD_POSITION, Queue = 0})
        return true
    end
    if action_name == "attack" then
        local target = self:NearestEnemy()
        if target == nil then return false end
        self:IssuePolicyOrder({UnitIndex = self.hero:entindex(), OrderType = DOTA_UNIT_ORDER_ATTACK_TARGET,
            TargetIndex = target:entindex(), Queue = 0})
        return true
    end
    -- Abilities and items remain masked until target-selection heads exist.
    return false
end

-- ExecuteOrderFromTable synchronously enters the order filter.  Marking this
-- narrow interval lets that filter retain a participant's real orders while
-- excluding only the policy's own commands.
function RLPPOBridge:IssuePolicyOrder(order)
    self.issuing_policy_order = true
    ExecuteOrderFromTable(order)
    self.issuing_policy_order = false
end

function RLPPOBridge:RequestAction()
    if self.request_in_flight or not self.enabled then return end
    local observation, mask, calibration = self:Observation()
    self.request_in_flight = true
    self:Post("/act", {observation = observation, action_mask = mask, game_time = GameRules:GetGameTime(),
        reward_version = self.reward_version, observation_version = "sf1v1_passive_v1",
        environment = self:EnvironmentMetadata()}, function(response)
        self.request_in_flight = false
        if response ~= nil then
            if self.enabled then self:Execute(response.action_name) end
            self.pending = response.decision_id
            self.pending_action = response.action_name
            self.pending_calibration = calibration
        end
        if self.finish_requested then self:FlushFinal() end
        if self.rotation_requested then self:FlushRotation() end
        if self.episode_end_requested then self:FinalizeEpisode() end
    end)
end

function RLPPOBridge:EndEpisode(after)
    self.enabled = false
    self.episode_end_requested = true
    self.episode_end_after = after
    self:FinalizeEpisode()
end

function RLPPOBridge:CompleteEpisode(ok)
    local after = self.episode_end_after
    self.episode_end_after = nil
    if after ~= nil then after(ok) end
end

function RLPPOBridge:FinalizeEpisode()
    if self.request_in_flight then return end
    if self.pending ~= nil then
        self:CommitPending(true, function(ok)
            self.episode_end_requested = false
            self:CompleteEpisode(ok)
        end)
    else
        self.episode_end_requested = false
        self:CompleteEpisode(true)
    end
end

function RLPPOBridge:IsEpisodeSettled()
    return not self.request_in_flight and self.pending == nil and not self.episode_end_requested
end

-- Source HTTP callbacks can occasionally be lost even after the localhost
-- bridge accepted the terminal transition.  Query the bridge before recovery:
-- a reset is permitted only when Python confirms that it has no pending sampled
-- decision, so an exact PPO transition is never discarded by this fallback.
function RLPPOBridge:RecoverEpisodeFromHealth(after)
    if self.episode_recovery_in_flight then return end
    self.episode_recovery_in_flight = true
    local request = CreateHTTPRequestScriptVM("GET", self.base_url .. "/health")
    request:SetHTTPRequestAbsoluteTimeoutMS(1000)
    request:Send(function(result)
        self.episode_recovery_in_flight = false
        if result.StatusCode ~= 200 then return end
        local response = json.decode(result.Body)
        if response ~= nil and tonumber(response.pending_decisions) == 0 then
            self.request_in_flight = false
            self.pending = nil
            self.episode_end_requested = false
            print("RL PPO bridge confirmed terminal settlement; recovering episode")
            self:CompleteEpisode(true)
        end
    end)
end

function RLPPOBridge:ResumeEpisode()
    if self.finish_requested then return end
    self.enabled = true
    self.previous_score = PlayerResource:GetGold(self.player_id) + 0.05 * self.hero:GetCurrentXP()
    self.episode_last_hits = 0
    self.calibration_last_hit = false
    self.calibration_hero_dead = false
    self:RequestAction()
end

function RLPPOBridge:CommitPending(done, after)
    if self.pending == nil then after(true); return end
    local pending = self.pending
    self.pending = nil
    self.request_in_flight = true
    local reward = self:Reward()
    self:EmitCalibration(pending, done, reward)
    self:Post("/transition", {decision_id = pending, reward = reward, done = done}, function(response)
        self.request_in_flight = false
        if self.finish_requested then
            self:FlushFinal()
            return
        end
        after(response ~= nil)
    end)
end

function RLPPOBridge:EmitCalibration(decision_id, done, reward)
    local payload = self.pending_calibration
    self.pending_calibration = nil
    if payload == nil then return end
    payload.decision_id = decision_id
    payload.action_name = self.pending_action or "idle"
    payload.done = done and true or false
    payload.reward = reward
    payload.last_hit = self.calibration_last_hit and true or false
    payload.hero_dead = self.calibration_hero_dead and true or false
    self.calibration_last_hit = false
    self.calibration_hero_dead = false
    self:Post("/calibration", payload, function(_) end)
end

function RLPPOBridge:Think()
    if not self.enabled or self.hero == nil or self.hero:IsNull() or not self.hero:IsAlive() then return self.decision_interval end
    if self.request_in_flight then return self.decision_interval end
    if self.pending ~= nil then
        self:CommitPending(false, function(ok)
            if ok and self.enabled then self:RequestAction() end
        end)
    else
        self:RequestAction()
    end
    return self.decision_interval
end

function RLPPOBridge:Finish()
    if self.finish_requested then return end
    self.enabled = false
    self.finish_requested = true
    self:FlushFinal()
end

-- Save one terminal batch, then continue using the same local bridge.  The
-- Python side rotates the filename when its --rollouts path contains {batch}.
function RLPPOBridge:Rotate(after)
    if self.finish_requested or self.rotation_requested then return end
    self.enabled = false
    self.rotation_requested = true
    self.rotation_after = after
    self:FlushRotation()
end

function RLPPOBridge:FlushRotation()
    if self.request_in_flight then return end
    if self.pending ~= nil then
        self:CommitPending(true, function(ok)
            if ok then self:FlushRotation()
            else self:CompleteRotation(false) end
        end)
        return
    end
    self.request_in_flight = true
    self:Post("/flush", {final_reward = self:Reward()}, function(response)
        self.request_in_flight = false
        if response ~= nil then print("Saved PPO rollout: " .. response.saved) end
        self:CompleteRotation(response ~= nil)
    end)
end

function RLPPOBridge:CompleteRotation(ok)
    self.rotation_requested = false
    local after = self.rotation_after
    self.rotation_after = nil
    if after ~= nil then after(ok) end
    if ok and after == nil and not self.finish_requested then self:ResumeEpisode() end
end

function RLPPOBridge:FlushFinal()
    if self.request_in_flight then return end
    if self.pending ~= nil then
        self:CommitPending(true, function(ok)
            if ok then self:FlushFinal() end
        end)
        return
    end
    self.request_in_flight = true
    -- `final_reward` also covers the tiny finish race where a just-sampled
    -- `/act` decision exists in Python but its callback has not yet updated
    -- Lua's `self.pending` field.
    self:Post("/flush", {final_reward = self:Reward()}, function(response)
        self.request_in_flight = false
        if response ~= nil then print("Saved PPO rollout: " .. response.saved) end
    end)
end

function RLPPOBridge:FirstOrderedUnit(filter)
    if filter.units == nil then return nil end
    for _, entindex in pairs(filter.units) do return EntIndexToHScript(entindex) end
    return nil
end

function RLPPOBridge:CastPayload(filter, payload)
    local unit = self:FirstOrderedUnit(filter)
    if unit == nil or filter.entindex_ability == nil then return nil end
    local ability = EntIndexToHScript(filter.entindex_ability)
    if ability == nil then return nil end
    for slot = 0, 5 do
        if unit:GetItemInSlot(slot) == ability then
            payload.order = "use_item"; payload.inventory_slot = slot + 1
            return payload
        end
    end
    -- This is the visible ability index. Custom games with hidden/passive
    -- abilities should supply their own stable ability-slot mapping here.
    for slot = 0, 5 do
        if unit:GetAbilityByIndex(slot) == ability then
            payload.order = "cast_ability"; payload.ability_slot = slot + 1
            return payload
        end
    end
    return nil
end

-- Called by a custom-game ExecuteOrderFilter. Dota's server sees semantic
-- orders but not the participant's physical key binding; raw_key is omitted.
function RLPPOBridge:LogHumanOrder(filter)
    local order = filter.order_type
    local player_id = filter.issuer_player_id_const
    local ordered_unit = self:FirstOrderedUnit(filter)
    -- Keep only manual orders from the selected participant's hero.  The
    -- policy marks its short ExecuteOrderFromTable interval, so its orders are
    -- excluded without also losing genuine player demonstrations.
    if self.hero == nil or ordered_unit == nil or ordered_unit:entindex() ~= self.hero:entindex() then return end
    if self.issuing_policy_order then return end
    local payload = {tick = math.floor(math.max(GameRules:GetGameTime(), 0) * 30), player_id = player_id}
    if order == DOTA_UNIT_ORDER_MOVE_TO_POSITION then
        payload.order = "move"; payload.target_kind = "position"; payload.target_x = filter.position_x; payload.target_y = filter.position_y
    elseif order == DOTA_UNIT_ORDER_ATTACK_TARGET then
        payload.order = "attack"; payload.target_kind = "entity"; payload.target_entity_id = filter.entindex_target
    elseif order == DOTA_UNIT_ORDER_CAST_TARGET then
        payload.target_kind = "entity"; payload.target_entity_id = filter.entindex_target; payload = self:CastPayload(filter, payload)
    elseif order == DOTA_UNIT_ORDER_CAST_POSITION then
        payload.target_kind = "position"; payload.target_x = filter.position_x; payload.target_y = filter.position_y; payload = self:CastPayload(filter, payload)
    elseif order == DOTA_UNIT_ORDER_CAST_NO_TARGET or order == DOTA_UNIT_ORDER_CAST_TOGGLE then
        payload = self:CastPayload(filter, payload)
    else
        return
    end
    if payload == nil then return end
    self:Post("/human-order", payload, function(_) end)
end
