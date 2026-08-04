-- Small, repeatable last-hit drill for local PPO collection.
-- It is intentionally a curriculum scenario, not a full Dota match.

if RLLaneScenario == nil then RLLaneScenario = class({}) end

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
    self.last_hits = 0
    self.waiting_for_terminal = false
    self:Reset()
end

function RLLaneScenario:RemoveCreeps()
    for _, creep in pairs(self.creeps) do
        if creep ~= nil and not creep:IsNull() then UTIL_Remove(creep) end
    end
    self.creeps = {}
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
    -- This is a fixed-state last-hit curriculum, not a full match. Reset
    -- progression so experience and gold cannot make later waves easier.
    if self.hero.SetHeroLevel ~= nil then self.hero:SetHeroLevel(1, true) end
    if self.hero.SetCurrentXP ~= nil then self.hero:SetCurrentXP(0) end
    if self.bridge.player_id ~= nil and PlayerResource.SetGold ~= nil then
        PlayerResource:SetGold(self.bridge.player_id, 0, false)
    end
    self.hero:SetHealth(self.hero:GetMaxHealth())
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
    elseif victim == self.hero then
        self.bridge:AddReward(-2.0, "hero_death")
        self:EndEpisode()
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
