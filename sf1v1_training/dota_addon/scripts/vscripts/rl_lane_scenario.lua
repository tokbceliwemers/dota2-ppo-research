-- Small, repeatable last-hit drill for local PPO collection.
-- It is intentionally a curriculum scenario, not a full Dota match.

if RLLaneScenario == nil then RLLaneScenario = class({}) end

local NECROMASTERY_ABILITY = "nevermore_necromastery"
local NECROMASTERY_MODIFIER = "modifier_nevermore_necromastery"
local PASSIVE_OPPONENT_NAME = "npc_dota_hero_nevermore"
local PASSIVE_OPPONENT_IDENTITY = "passive_nevermore_v1"

local function enemy_team(team)
    return team == DOTA_TEAM_GOODGUYS and DOTA_TEAM_BADGUYS or DOTA_TEAM_GOODGUYS
end

local function creep_names(team)
    if team == DOTA_TEAM_GOODGUYS then
        return "npc_dota_creep_goodguys_melee", "npc_dota_creep_goodguys_ranged"
    end
    return "npc_dota_creep_badguys_melee", "npc_dota_creep_badguys_ranged"
end

function RLLaneScenario:Start(hero, bridge)
    self.hero = hero
    self.bridge = bridge
    self.anchor = hero:GetAbsOrigin()
    self.episode_seconds = 75
    self.creeps = {}
    self.enemy_creeps = {}
    self.opponent = nil
    self.last_hits = 0
    self.waiting_for_terminal = false
    self:Reset()
end

function RLLaneScenario:RemoveOpponent()
    if self.opponent ~= nil and not self.opponent:IsNull() then UTIL_Remove(self.opponent) end
    self.opponent = nil
    self.bridge:SetOpponent(nil, "absent", "none")
end

function RLLaneScenario:Destroy()
    self:RemoveCreeps()
    self:RemoveOpponent()
end

function RLLaneScenario:RemoveCreeps()
    for _, creep in pairs(self.creeps) do
        if creep ~= nil and not creep:IsNull() then UTIL_Remove(creep) end
    end
    self.creeps = {}
end

function RLLaneScenario:ClearNecromastery(unit)
    -- Necromastery stacks change attack damage, so letting them survive a
    -- wave would make later episodes easier than earlier ones.  Keep this
    -- micro-curriculum about movement and last-hit timing, not soul farming.
    unit = unit or self.hero
    local ability = unit:FindAbilityByName(NECROMASTERY_ABILITY)
    if ability ~= nil and ability:GetLevel() ~= 0 then ability:SetLevel(0) end
    local modifier = unit:FindModifierByName(NECROMASTERY_MODIFIER)
    if modifier ~= nil then modifier:SetStackCount(0) end
    unit:RemoveModifierByName(NECROMASTERY_MODIFIER)
end

function RLLaneScenario:NormalizeHeroProgression(unit)
    -- The experience filter is the authoritative guard.  These assignments
    -- also repair any XP or ability state that existed before a reset.
    unit = unit or self.hero
    if unit.SetCurrentXP ~= nil and unit:GetCurrentXP() ~= 0 then unit:SetCurrentXP(0) end
    if unit.SetAbilityPoints ~= nil then unit:SetAbilityPoints(0) end
    self:ClearNecromastery(unit)
end

function RLLaneScenario:ResetPassiveOpponent()
    local team = enemy_team(self.hero:GetTeamNumber())
    local position = self.anchor + Vector(0, 350, 0)
    if self.opponent == nil or self.opponent:IsNull() or not self.opponent:IsAlive() then
        self:RemoveOpponent()
        self.opponent = CreateUnitByName(PASSIVE_OPPONENT_NAME, position, true, nil, nil, team)
    end
    FindClearSpaceForUnit(self.opponent, position, true)
    if self.opponent.SetHeroLevel ~= nil then self.opponent:SetHeroLevel(1, false) end
    self:NormalizeHeroProgression(self.opponent)
    self.opponent:SetHealth(self.opponent:GetMaxHealth())
    self.opponent:SetMana(self.opponent:GetMaxMana())
    ExecuteOrderFromTable({UnitIndex = self.opponent:entindex(), OrderType = DOTA_UNIT_ORDER_HOLD_POSITION, Queue = 0})
    self.bridge:SetOpponent(self.opponent, "passive", PASSIVE_OPPONENT_IDENTITY)
end

function RLLaneScenario:PrintOpponent()
    if self.opponent == nil or self.opponent:IsNull() then
        print("RL PPO opponent: unavailable")
        return
    end
    local origin = self.opponent:GetAbsOrigin()
    print(string.format("RL PPO opponent: mode=passive identity=%s team=%d hp=%d/%d mana=%d/%d x=%.0f y=%.0f",
        PASSIVE_OPPONENT_IDENTITY, self.opponent:GetTeamNumber(), self.opponent:GetHealth(), self.opponent:GetMaxHealth(),
        self.opponent:GetMana(), self.opponent:GetMaxMana(), origin.x, origin.y))
end

function RLLaneScenario:SpawnWave(team, origin, direction, tracked_enemy_wave)
    local melee, ranged = creep_names(team)
    local names = {melee, melee, melee, ranged}
    for index, name in ipairs(names) do
        local offset = direction * (index * 70)
        local creep = CreateUnitByName(name, origin + offset, true, nil, nil, team)
        table.insert(self.creeps, creep)
        if tracked_enemy_wave then self.enemy_creeps[creep:entindex()] = true end
        ExecuteOrderFromTable({UnitIndex = creep:entindex(), OrderType = DOTA_UNIT_ORDER_ATTACK_MOVE,
            Position = self.anchor, Queue = 0})
    end
end

function RLLaneScenario:Reset()
    self:RemoveCreeps()
    if not self.hero:IsAlive() then self.hero:RespawnHero(false, false) end
    FindClearSpaceForUnit(self.hero, self.anchor, true)
    -- This is a fixed-state last-hit curriculum, not a full match. A fresh
    -- lobby starts at Level 1; the experience filter prevents later levels.
    if self.hero.SetCurrentXP ~= nil then self.hero:SetCurrentXP(0) end
    if self.hero.SetHeroLevel ~= nil then self.hero:SetHeroLevel(1, false) end
    if self.bridge.player_id ~= nil and PlayerResource.SetGold ~= nil then
        PlayerResource:SetGold(self.bridge.player_id, 0, false)
    end
    self:NormalizeHeroProgression()
    self.hero:SetHealth(self.hero:GetMaxHealth())
    self.hero:SetMana(self.hero:GetMaxMana())
    self:ResetPassiveOpponent()
    self.last_hits = 0
    self.enemy_creeps = {}
    self.episode_started = GameRules:GetGameTime()
    self.waiting_for_terminal = false
    local team = self.hero:GetTeamNumber()
    self:SpawnWave(team, self.anchor + Vector(-550, 0, 0), Vector(1, 0, 0), false)
    self:SpawnWave(enemy_team(team), self.anchor + Vector(550, 0, 0), Vector(-1, 0, 0), true)
    self.bridge:ResumeEpisode()
end

function RLLaneScenario:EnemyWaveCleared()
    for entindex, tracked in pairs(self.enemy_creeps) do
        if tracked then
            local creep = EntIndexToHScript(entindex)
            if creep ~= nil and not creep:IsNull() and creep:IsAlive() then return false end
        end
    end
    return true
end

function RLLaneScenario:OnEntityKilled(event)
    local victim = event.entindex_killed and EntIndexToHScript(event.entindex_killed) or nil
    local attacker = event.entindex_attacker and EntIndexToHScript(event.entindex_attacker) or nil
    if victim == nil then return end
    local tracked_enemy = victim:IsCreep() and self.enemy_creeps[victim:entindex()] == true
    if tracked_enemy then self.enemy_creeps[victim:entindex()] = false end
    if attacker == self.hero and victim:IsCreep() then
        self.last_hits = self.last_hits + 1
        self.bridge:AddReward(1.0, "last_hit")
        self:ClearNecromastery()
    elseif victim == self.hero then
        self.bridge:AddReward(-2.0, "hero_death")
        self:EndEpisode()
    elseif victim == self.opponent then
        self:EndEpisode("passive_opponent_death")
    end
    if tracked_enemy and self:EnemyWaveCleared() then self:EndEpisode("enemy_wave_cleared") end
end

function RLLaneScenario:EndEpisode(reason)
    if self.waiting_for_terminal then return end
    self.waiting_for_terminal = true
    self.terminal_requested_at = GameRules:GetGameTime()
    if reason ~= nil then print("RL PPO lane episode complete: " .. reason) end
    self.bridge:AddReward(0.05 * self.last_hits)
    self.bridge:EndEpisode(function(ok)
        if ok then
            print("RL PPO lane episode committed; starting fresh wave")
            self:Reset()
        else
            print("RL PPO lane episode commit failed; use rl_ppo_restart after checking the Python bridge")
        end
    end)
end

function RLLaneScenario:Think()
    self:NormalizeHeroProgression()
    if self.waiting_for_terminal then
        -- The normal path resets from the terminal HTTP callback.  This is a
        -- safe recovery for a missed callback: only reset after the bridge
        -- confirms that no sampled decision remains in flight or uncommitted.
        if self.bridge:IsEpisodeSettled() then
            print("RL PPO lane terminal settled without callback; recovering fresh wave")
            self:Reset()
        elseif GameRules:GetGameTime() - self.terminal_requested_at >= 2.0 then
            self.bridge:RecoverEpisodeFromHealth(function(_) end)
        end
        return
    end
    if GameRules:GetGameTime() - self.episode_started >= self.episode_seconds then self:EndEpisode() end
end
