"""Sensor index convention"""


cdef enum:

    # Links
    LINK_SIZE = 20
    LINK_COM_POSITION_X = 0
    LINK_COM_POSITION_Y = 1
    LINK_COM_POSITION_Z = 2
    LINK_COM_ORIENTATION_X = 3
    LINK_COM_ORIENTATION_Y = 4
    LINK_COM_ORIENTATION_Z = 5
    LINK_COM_ORIENTATION_W = 6
    LINK_URDF_POSITION_X = 7
    LINK_URDF_POSITION_Y = 8
    LINK_URDF_POSITION_Z = 9
    LINK_URDF_ORIENTATION_X = 10
    LINK_URDF_ORIENTATION_Y = 11
    LINK_URDF_ORIENTATION_Z = 12
    LINK_URDF_ORIENTATION_W = 13
    LINK_COM_VELOCITY_LIN_X = 14
    LINK_COM_VELOCITY_LIN_Y = 15
    LINK_COM_VELOCITY_LIN_Z = 16
    LINK_COM_VELOCITY_ANG_X = 17
    LINK_COM_VELOCITY_ANG_Y = 18
    LINK_COM_VELOCITY_ANG_Z = 19

    # Joints
    JOINT_SIZE = 16
    JOINT_POSITION = 0
    JOINT_VELOCITY = 1
    JOINT_TORQUE = 2
    JOINT_FORCE_X = 3
    JOINT_FORCE_Y = 4
    JOINT_FORCE_Z = 5
    JOINT_TORQUE_X = 6
    JOINT_TORQUE_Y = 7
    JOINT_TORQUE_Z = 8
    JOINT_CMD_POSITION = 9
    JOINT_CMD_VELOCITY = 10
    JOINT_CMD_TORQUE = 11
    JOINT_TORQUE_ACTIVE = 12
    JOINT_TORQUE_STIFFNESS = 13
    JOINT_TORQUE_DAMPING = 14
    JOINT_TORQUE_FRICTION = 15

    # Contacts
    CONTACT_SIZE = 12
    CONTACT_REACTION_X = 0
    CONTACT_REACTION_Y = 1
    CONTACT_REACTION_Z = 2
    CONTACT_FRICTION_X = 3
    CONTACT_FRICTION_Y = 4
    CONTACT_FRICTION_Z = 5
    CONTACT_TOTAL_X = 6
    CONTACT_TOTAL_Y = 7
    CONTACT_TOTAL_Z = 8
    CONTACT_POSITION_X = 9
    CONTACT_POSITION_Y = 10
    CONTACT_POSITION_Z = 11

    # Xfrc
    XFRC_SIZE = 6
    XFRC_FORCE_X = 0
    XFRC_FORCE_Y = 1
    XFRC_FORCE_Z = 2
    XFRC_TORQUE_X = 3
    XFRC_TORQUE_Y = 4
    XFRC_TORQUE_Z = 5

    # Muscles
    MUSCLE_SIZE = 11
    MUSCLE_ACTIVATION = 0
    # MTU
    MUSCLE_TENDON_UNIT_LENGTH = 1
    MUSCLE_TENDON_UNIT_VELOCITY = 2
    MUSCLE_TENDON_UNIT_FORCE = 3
    # MUSCLE-FIBER
    MUSCLE_FIBER_LENGTH = 4
    MUSCLE_FIBER_VELOCITY = 6
    MUSCLE_PENNATION_ANGLE = 7
    MUSCLE_ACTIVE_FORCE = 8
    MUSCLE_PASSIVE_FORCE = 9
    # TENDON
    MUSCLE_TENDON_LENGTH = 10
    MUSCLE_TENDON_FORCE = 11
    # FEEDBACKS
    MUSCLE_IA_FEEDBACK = 12
    MUSCLE_II_FEEDBACK = 13
    MUSCLE_IB_FEEDBACK = 14
