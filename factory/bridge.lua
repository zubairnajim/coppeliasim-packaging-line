-- FactoryBridge: in-simulator helper for the external Python cell controller.
--  * solves inverse kinematics for every robot each step (targets are plain dummies)
--  * applies batched pose updates sent by Python in a single remote call
sim = require 'sim'
simIK = require 'simIK'

local ARMS = {'UR10', 'IRB4600', 'BaxterL', 'BaxterR'}

function sysCall_init()
    ikEnv = simIK.createEnvironment()
    arms = {}
    for _, n in ipairs(ARMS) do
        local base = sim.getObject('/' .. n .. '_ikBase', {noError = true})
        if base ~= -1 then
            local g = simIK.createGroup(ikEnv)
            simIK.setGroupCalculation(ikEnv, g, simIK.method_damped_least_squares, 0.03, 40)
            simIK.addElementFromScene(ikEnv, g, sim.getObjectParent(base), sim.getObject('/' .. n .. '_tip'),
                                      sim.getObject('/' .. n .. '_target'), simIK.constraint_pose)
            arms[#arms + 1] = {name = n, group = g, target = sim.getObject('/' .. n .. '_target')}
        end
    end
    ikFail = {}
    active = {}      -- target handle -> steps of IK still to run (targets that did not move are skipped)
    initDelta()
end

function initDelta()
    local base = sim.getObject('/irb360', {noError = true})
    if base == -1 then return end
    local function o(p) return sim.getObject('/irb360/' .. p) end
    local d = {target = sim.getObject('/Delta_target')}
    d.cart = {o('cartesianX'), o('cartesianY'), o('cartesianZ')}
    d.motor = o('motor')
    d.ikTarget = o('ikTarget')
    d.p0 = sim.getObjectPosition(d.ikTarget)
    d.axes = {}
    for i = 1, 3 do
        local m = sim.getObjectMatrix(d.cart[i])
        d.axes[i] = {m[3], m[7], m[11]}
    end
    local tp = sim.getObjectPosition(sim.getObject('/Delta_tip'))
    d.tcpOffset = {tp[1] - d.p0[1], tp[2] - d.p0[2], tp[3] - d.p0[3]}

    d.main = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, d.main, simIK.method_pseudo_inverse, 0, 6)
    simIK.addElementFromScene(ikEnv, d.main, base, o('ikTip'), d.ikTarget, simIK.constraint_position)
    for _, s in ipairs({'', '0', '1', '2', '3'}) do
        simIK.addElementFromScene(ikEnv, d.main, base, o('loopTip' .. s), o('loopTarget' .. s), simIK.constraint_pose)
    end
    d.axis = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, d.axis, simIK.method_pseudo_inverse, 0, 6)
    local e = simIK.addElementFromScene(ikEnv, d.axis, o('axisL'), o('axisTip'), o('axisTarget'), simIK.constraint_pose)
    simIK.setElementPrecision(ikEnv, d.axis, e, {0.0001, 0.1})
    d.bridges = {}
    local br = {{'j2', 'L', ''}, {'j26', 'R', ''}, {'j28', 'L', '0'}, {'j29', 'R', '0'}, {'j31', 'L', '1'}, {'j34', 'R', '1'}}
    for i, b in ipairs(br) do
        local g = simIK.createGroup(ikEnv)
        simIK.setGroupCalculation(ikEnv, g, simIK.method_damped_least_squares, 0.01, 3)
        simIK.addElementFromScene(ikEnv, g, o(b[1]), o('bridge' .. b[2] .. 'Tip' .. b[3]),
                                  o('bridge' .. b[2] .. 'Target' .. b[3]), simIK.constraint_position)
        d.bridges[i] = g
    end
    delta = d
end

function stepDelta()
    local d = delta
    local pose = sim.getObjectPose(d.target)
    local yaw = sim.getObjectOrientation(d.target)[3]
    for i = 1, 3 do
        local q = 0
        for k = 1, 3 do q = q + (pose[k] - d.tcpOffset[k] - d.p0[k]) * d.axes[i][k] end
        sim.setJointPosition(d.cart[i], q)
    end
    sim.setJointPosition(d.motor, yaw)
    local ok = simIK.handleGroup(ikEnv, d.main, {syncWorlds = true}) == simIK.result_success
    ok = ok and simIK.handleGroup(ikEnv, d.axis, {syncWorlds = true}) == simIK.result_success
    if ok then
        for i = 1, #d.bridges do simIK.handleGroup(ikEnv, d.bridges[i], {syncWorlds = true, allowError = true}) end
    else
        ikFail['Delta'] = (ikFail['Delta'] or 0) + 1
    end
end

function sysCall_actuation()
    if delta and (active[delta.target] or 0) > 0 then
        active[delta.target] = active[delta.target] - 1
        stepDelta()
    end
    for _, a in ipairs(arms) do
        local n = active[a.target] or 0
        if n > 0 then active[a.target] = n - 1 end
        if n > 0 and simIK.handleGroup(ikEnv, a.group, {syncWorlds = true, allowError = true}) ~= simIK.result_success then
            ikFail[a.name] = (ikFail[a.name] or 0) + 1
        end
    end
end

-- handles: {h1,h2,...}; poses: flat {x,y,z,qx,qy,qz,qw, ...} in world frame
function setPoses(handles, poses)
    for i = 1, #handles do
        local k = (i - 1) * 7
        sim.setObjectPose(handles[i], {poses[k + 1], poses[k + 2], poses[k + 3], poses[k + 4], poses[k + 5], poses[k + 6], poses[k + 7]})
        active[handles[i]] = 3
    end
end

function getIkFailures()
    return ikFail
end

function sysCall_cleanup()
    simIK.eraseEnvironment(ikEnv)
end
