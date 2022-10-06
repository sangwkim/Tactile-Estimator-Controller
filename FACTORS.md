At the time of writing, there are 11 custom factors that are used in the project. 
```
DispDiff
PoseDiff
PenHinge
ContactMotion
PenEven
DispVar
EnergyElastic
TorqPoint
TorqLine
Wrench
WrenchInc
```
The source code of these factors can be found at 'cpp/' folder.

## DispDiff

This factor impose a relative motion from one timestep to another. 

There are 4 Input Variables and 1 Measurement Pose:
1. Reference Pose at Reference Timestep (P1)
2. Reference Pose at Current Timestep (P2)
3. Another Pose at Reference Timestep (P3)
4. Another Pose at Current Timestep (P4)

In other words, it imposes: ((P1)^(-1)*(P3))^(-1)*((P2)^(-1)*(P4)) == (Measurement)

## PoseDiff

This factor impose that the last Pose variable (P3) be equal to the Pose difference between the first (P1) and second (P2) Pose variable.

In other words, it imposes: (P1)^(-1)*(P2) == (P3)

## PenHinge

This factor impose the minimum penetration distance of contact through the object.

There are 3 Input Variables:
1. Undeformed Object Pose (Pn)
2. Estimated Contact (Pc)
3. Deformed Object Pose (Po)

It first computes the penetration distance by taking the z-component of the following Pose: ((Pn)^(-1)*(Pc))^(-1)*((Po)^(-1)*(Pc))

Then, it impose an inequality relation (penetration distance) >= (minimum distance) by using hinge loss function (implemented by if statement in the code).

## ContactMotion

This factor impose that the control input is equal to the motion at the estimated contact point between consecutive timesteps.

There are 4 Input Variables:
1. Gripper Pose at previous timestep
2. Gripper Pose at next timestep
3. Estimated Contact
4. Control Input

In other words, the control input is the same as the local motion of the gripper at the estimated contact point.

## PenEven

This factor is imposed when we want the penetration to be evenly distribution along the contact line or contact patch.

## DispVar, EnergyElastic, TorqPoint, TorqLine, Wrench, WrenchInc (description to be added)
