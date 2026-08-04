-- Reference entry point. Merge this into your own local custom-game mode.
require("rl_ppo_bridge")
require("rl_lane_scenario")

if GameMode == nil then GameMode = class({}) end

local RL_BATCH_SECONDS = 160
local RL_PLAYER_ID = 0
local RL_TRAINING_HERO = "npc_dota_hero_nevermore"

function Activate()
    GameRules.GameMode = GameMode()
    GameRules.GameMode:InitGameMode()
end

function GameMode:InitGameMode()
    -- One second is intentional: it keeps the direct startup fully automatic
    -- while guaranteeing a selection-state callback for MakeRandomHeroSelection.
    GameRules:SetHeroSelectionTime(1)
    GameRules:SetStrategyTime(0)
    GameRules:SetShowcaseTime(0)
    GameRules:SetPreGameTime(0)
    self.rl_bridge = RLPPOBridge()
    self.rl_scenario = RLLaneScenario()
    GameRules:GetGameModeEntity():SetExecuteOrderFilter(Dynamic_Wrap(GameMode, "FilterExecuteOrder"), self)
    GameRules:GetGameModeEntity():SetModifyExperienceFilter(Dynamic_Wrap(GameMode, "FilterModifyExperience"), self)
    ListenToGameEvent("npc_spawned", Dynamic_Wrap(GameMode, "OnNPCSpawned"), self)
    ListenToGameEvent("game_rules_state_change", Dynamic_Wrap(GameMode, "OnGameStateChanged"), self)
    ListenToGameEvent("entity_killed", Dynamic_Wrap(GameMode, "OnEntityKilled"), self)
    Convars:RegisterCommand("rl_ppo_finish", function()
        if GameRules.GameMode ~= nil and GameRules.GameMode.rl_bridge ~= nil then
            GameRules.GameMode:StopBatchTimer()
            GameRules.GameMode.rl_bridge:Finish()
        end
    end, "End the local RL episode and save its PPO rollout", 0)
    Convars:RegisterCommand("rl_ppo_timer", function()
        if GameRules.GameMode ~= nil then GameRules.GameMode:PrintBatchTimer() end
    end, "Local-only: print PPO batch timer status", 0)
    Convars:RegisterCommand("rl_ppo_progress", function()
        if GameRules.GameMode ~= nil then GameRules.GameMode:PrintRLProgress() end
    end, "Local-only: print fixed-progression curriculum state", 0)
    Convars:RegisterCommand("rl_ppo_speed_2", function()
        if GameRules.GameMode ~= nil then GameRules.GameMode:SetRLPPOTimeScale(2) end
    end, "Local-only: run the lane curriculum at 2x simulation speed", 0)
    Convars:RegisterCommand("rl_ppo_speed_4", function()
        if GameRules.GameMode ~= nil then GameRules.GameMode:SetRLPPOTimeScale(4) end
    end, "Local-only: run the lane curriculum at 4x simulation speed", 0)
    Convars:RegisterCommand("rl_ppo_speed_reset", function()
        if GameRules.GameMode ~= nil then GameRules.GameMode:SetRLPPOTimeScale(1) end
    end, "Local-only: restore normal simulation speed", 0)
    Convars:RegisterCommand("rl_ppo_restart", function()
        local mode = GameRules.GameMode
        if mode == nil or not mode:RestartRLPPO() then
            print("RL PPO restart: no controlled hero is available")
        end
    end, "Local-only: reconnect a fresh Python bridge and start a new lane batch", 0)
    Convars:RegisterCommand("rl_ppo_test_attack", function()
        if GameRules.GameMode ~= nil and GameRules.GameMode.rl_bridge ~= nil then
            if not GameRules.GameMode.rl_bridge:Execute("attack") then
                print("RL PPO attack test: no enemy creep within 800 range")
            else
                print("RL PPO attack test: attacking nearest enemy creep")
            end
        end
    end, "Local-only: test attack targeting against the nearest enemy creep", 0)
    Convars:RegisterCommand("rl_ppo_test_last_hit", function()
        local mode = GameRules.GameMode
        if mode == nil or mode.rl_bridge == nil then return end
        local target = mode.rl_bridge:NearestEnemy()
        if target == nil then
            print("RL PPO last-hit test: no enemy creep within 800 range")
            return
        end
        -- A local test probe for the real entity_killed -> +1 reward route.
        -- Normal curriculum rewards continue to come from ordinary combat.
        ApplyDamage({victim = target, attacker = mode.rl_bridge.hero, damage = target:GetHealth() + 1,
            damage_type = DAMAGE_TYPE_PURE, damage_flags = DOTA_DAMAGE_FLAG_BYPASSES_INVULNERABILITY})
        print("RL PPO last-hit test: forced one hero last hit")
    end, "Local-only: verify the shaped last-hit reward callback", 0)
end

function GameMode:StartBatchTimer()
    self.rl_batch_started_at = GameRules:GetGameTime()
    self.rl_batch_timer_active = true
    self.rl_batch_30_second_notice_sent = false
    local real_seconds = math.ceil(RL_BATCH_SECONDS / (self.rl_time_scale or 1))
    print(string.format("RL PPO batch timer started: save after %d game seconds (~%d real seconds at %gx)",
        RL_BATCH_SECONDS, real_seconds, self.rl_time_scale or 1))
end

function GameMode:SetRLPPOTimeScale(multiplier)
    multiplier = math.max(1, math.min(4, tonumber(multiplier) or 1))
    self.rl_time_scale = multiplier
    -- host_timescale is a cheat-protected local-server ConVar.  The custom
    -- lobby must have Enable Cheats checked; never use this outside the local
    -- training addon.  Game-time timers and PPO decision cadence stay intact.
    SendToServerConsole("host_timescale " .. multiplier)
    local real_seconds = math.ceil(RL_BATCH_SECONDS / multiplier)
    print(string.format("RL PPO speed requested: %gx; one batch is %d game seconds (~%d real seconds)",
        multiplier, RL_BATCH_SECONDS, real_seconds))
end

function GameMode:StopBatchTimer()
    self.rl_batch_timer_active = false
end

function GameMode:PrintBatchTimer()
    if not self.rl_batch_timer_active or self.rl_batch_started_at == nil then
        print("RL PPO batch timer is not active")
        return
    end
    local elapsed = math.max(GameRules:GetGameTime() - self.rl_batch_started_at, 0)
    local remaining = math.max(RL_BATCH_SECONDS - elapsed, 0)
    print(string.format("RL PPO batch timer: %d seconds elapsed, %d seconds remaining", math.floor(elapsed), math.ceil(remaining)))
end

function GameMode:PrintRLProgress()
    local hero = self.rl_bridge ~= nil and self.rl_bridge.hero or nil
    if hero == nil or hero:IsNull() then
        print("RL PPO progression: no controlled hero is available")
        return
    end
    local ability = hero:FindAbilityByName("nevermore_necromastery")
    local modifier = hero:FindModifierByName("modifier_nevermore_necromastery")
    local ability_level = ability ~= nil and ability:GetLevel() or 0
    local stacks = modifier ~= nil and modifier:GetStackCount() or 0
    local gold = self.rl_bridge.player_id ~= nil and PlayerResource:GetGold(self.rl_bridge.player_id) or 0
    print(string.format("RL PPO progression: level=%d xp=%d gold=%d necromastery_level=%d necromastery_stacks=%d",
        hero:GetLevel(), math.floor(hero:GetCurrentXP()), gold, ability_level, stacks))
end

function GameMode:BatchTimerThink()
    if not self.rl_batch_timer_active or self.rl_batch_started_at == nil then return end
    local elapsed = GameRules:GetGameTime() - self.rl_batch_started_at
    if elapsed >= RL_BATCH_SECONDS then
        self:StopBatchTimer()
        print("RL PPO batch timer complete: 160 game seconds elapsed; saving rollout")
        self.rl_bridge:Finish()
    elseif elapsed >= RL_BATCH_SECONDS - 30 and not self.rl_batch_30_second_notice_sent then
        self.rl_batch_30_second_notice_sent = true
        print("RL PPO batch timer: 30 seconds until automatic save")
    end
end

function GameMode:RestartRLPPO()
    local hero = self.rl_bridge ~= nil and self.rl_bridge.hero or nil
    if hero == nil or hero:IsNull() then return false end
    if self.rl_scenario ~= nil then self.rl_scenario:RemoveCreeps() end
    self.rl_bridge = RLPPOBridge()
    self.rl_scenario = RLLaneScenario()
    self.rl_bridge:Start(hero, RL_PLAYER_ID)
    self.rl_scenario:Start(hero, self.rl_bridge)
    self:StartBatchTimer()
    print("RL PPO restart: new lane batch started")
    return true
end

function GameMode:OnNPCSpawned(event)
    local hero = EntIndexToHScript(event.entindex)
    if self.rl_started then return end
    if hero ~= nil and hero:IsRealHero() and hero:GetPlayerOwnerID() == RL_PLAYER_ID then
        if hero:GetUnitName() ~= RL_TRAINING_HERO then
            if self.rl_replacing_hero then return end
            self.rl_replacing_hero = true
            PrecacheUnitByNameAsync(RL_TRAINING_HERO, function()
                if PlayerResource:IsValidPlayerID(RL_PLAYER_ID) then
                    PlayerResource:ReplaceHeroWith(RL_PLAYER_ID, RL_TRAINING_HERO, 0, 0)
                end
            end, RL_PLAYER_ID)
            return
        end
        self.rl_started = true
        self.rl_bridge:Start(hero, RL_PLAYER_ID)
        self.rl_scenario:Start(hero, self.rl_bridge)
        self:StartBatchTimer()
        GameRules:GetGameModeEntity():SetThink(function()
            return GameRules.GameMode:RLPPOThink()
        end, "rl_ppo_think", 0.25)
    end
end

function GameMode:RLPPOThink()
    if self.rl_bridge == nil then return nil end
    self:BatchTimerThink()
    if self.rl_scenario ~= nil then self.rl_scenario:Think() end
    return self.rl_bridge:Think()
end

function GameMode:OnEntityKilled(event)
    if self.rl_scenario ~= nil then self.rl_scenario:OnEntityKilled(event) end
end

function GameMode:OnGameStateChanged()
    local state = GameRules:State_Get()
    -- A startup command can compress the selection states to zero seconds.
    -- Select as soon as either selection callback is observed; the spawned
    -- placeholder is immediately replaced with the fixed local Shadow Fiend.
    if state == DOTA_GAMERULES_STATE_HERO_SELECTION or state == DOTA_GAMERULES_STATE_STRATEGY_TIME then
        local player = PlayerResource:GetPlayer(RL_PLAYER_ID)
        if player ~= nil and not PlayerResource:HasSelectedHero(RL_PLAYER_ID) then
            player:MakeRandomHeroSelection()
        end
    end
    if self.rl_started and state == DOTA_GAMERULES_STATE_POST_GAME then
        self:StopBatchTimer()
        self.rl_bridge:Finish()
    end
end

function GameMode:FilterExecuteOrder(filter)
    if self.rl_bridge ~= nil then self.rl_bridge:LogHumanOrder(filter) end
    return true
end

function GameMode:FilterModifyExperience(filter)
    -- Do not allow creep XP to promote the controlled hero during a fixed
    -- Level-1 last-hit drill.  Other entities keep normal custom-lobby XP.
    local hero = self.rl_bridge ~= nil and self.rl_bridge.hero or nil
    if hero ~= nil and not hero:IsNull() and filter["hero_entindex_const"] == hero:entindex() then
        return false
    end
    return true
end
