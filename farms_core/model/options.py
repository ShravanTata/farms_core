"""Animat options"""

from enum import Enum, IntEnum
from typing import List, Dict, Union
from ..options import Options


class SpawnLoader(IntEnum):
    """Spawn loader"""
    FARMS = 0
    PYBULLET = 1


class MorphologyOptions(Options):
    """Morphology options"""

    def __init__(self, **kwargs):
        super().__init__()
        links = kwargs.pop('links')
        self.links: List[LinkOptions] = (
            links
            if all(isinstance(link, LinkOptions) for link in links)
            else [LinkOptions(**link) for link in links]
        )
        self.self_collisions: List[List[str]] = kwargs.pop('self_collisions')
        joints = kwargs.pop('joints')
        self.joints: List[JointOptions] = (
            joints
            if all(isinstance(joint, JointOptions) for joint in joints)
            else [JointOptions(**joint) for joint in joints]
        )
        tendons = kwargs.pop('tendons', [])
        self.tendons: List[TendonOptions] = (
            tendons
            if all(isinstance(tendon, TendonOptions) for tendon in tendons)
            else [TendonOptions.from_options(**tendon) for tendon in tendons]
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    def links_names(self) -> List[str]:
        """Links names"""
        return [link.name for link in self.links]

    def joints_names(self) -> List[str]:
        """Joints names"""
        return [joint.name for joint in self.joints]

    def tendons_names(self) -> List[str]:
        """Tendons names"""
        return [tendon.name for tendon in self.tendons]

    def n_joints(self) -> int:
        """Number of joints"""
        return len(self.joints)

    def n_links(self) -> int:
        """Number of links"""
        return len(self.links)

    def n_tendons(self) -> int:
        """ Number of tendons """
        return len(self.tendons)


class LinkOptions(Options):
    """Link options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.collisions: bool = kwargs.pop('collisions')
        self.friction: List[float] = kwargs.pop('friction')
        self.sites: List[SiteOptions] = kwargs.pop('sites', [])
        self.extras: Dict = kwargs.pop('extras', {})
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class JointOptions(Options):
    """Joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.initial: List[float] = kwargs.pop('initial')
        self.limits: List[float] = kwargs.pop('limits')
        self.stiffness: float = kwargs.pop('stiffness')
        self.springref: float = kwargs.pop('springref')
        self.damping: float = kwargs.pop('damping')
        self.extras: Dict = kwargs.pop('extras', {})
        for i, state in enumerate(['position', 'velocity']):
            assert self.limits[i][0] <= self.limits[i][1], (
                f'Minimum must be smaller than maximum for {state} limits'
            )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class SpawnOptions(Options):
    """Spawn options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.loader: SpawnLoader = kwargs.pop('loader')
        self.pose: List[float] = [*kwargs.pop('pose')]
        self.velocity: List[float] = [*kwargs.pop('velocity')]
        assert len(self.pose) == 6
        assert len(self.velocity) == 6
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        options = {}
        # Loader
        options['loader'] = kwargs.pop('spawn_loader', SpawnLoader.FARMS)
        options['pose'] = (
            # Position in [m]
            list(kwargs.pop('spawn_position', [0, 0, 0]))
            # Orientation in [rad] (Euler angles)
            + list(kwargs.pop('spawn_orientation', [0, 0, 0]))
        )
        options['velocity'] = (
            # Linear velocity in [m/s]
            list(kwargs.pop('spawn_velocity_lin', [0, 0, 0]))
            # Angular velocity in [rad/s]
            + list(kwargs.pop('spawn_velocity_ang', [0, 0, 0]))
        )
        return cls(**options)


class ControlOptions(Options):
    """Control options"""

    def __init__(self, **kwargs):
        super().__init__()
        sensors = kwargs.pop('sensors')
        self.sensors: SensorsOptions = (
            sensors
            if isinstance(sensors, SensorsOptions)
            else SensorsOptions(**sensors)
        )
        motors = kwargs.pop('motors')
        self.motors: List[MotorOptions] = (
            motors
            if all(isinstance(motor, MotorOptions) for motor in motors)
            else [MotorOptions(**motor) for motor in motors]
        )
        muscles = kwargs.pop('hill_muscles', [])
        self.hill_muscles: List[MuscleOptions] = (
            muscles
            if all(isinstance(muscle, MuscleOptions) for muscle in muscles)
            else [MuscleOptions(**muscle) for muscle in muscles]
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['sensors'] = kwargs.pop(
            'sensors',
            SensorsOptions.from_options(kwargs).to_dict()
        )
        options['motors'] = kwargs.pop('motors', [])
        options['muscles'] = kwargs.pop('muscles', [])
        return options

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))

    def joints_names(self) -> List[str]:
        """Joints names"""
        return [motor.joint_name for motor in self.motors]

    def motors_limits_torque(self) -> List[float]:
        """Motors max torques"""
        return [motor.limits_torque for motor in self.motors]


class MotorOptions(Options):
    """Motor options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.control_types: List[str] = kwargs.pop('control_types')
        self.limits_torque: List[float] = kwargs.pop('limits_torque')
        self.gains: List[float] = kwargs.pop('gains')
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class SensorsOptions(Options):
    """Sensors options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.links: List[str] = kwargs.pop('links')
        self.joints: List[str] = kwargs.pop('joints')
        self.contacts: List[List[str]] = kwargs.pop('contacts')
        self.xfrc: List[str] = kwargs.pop('xfrc')
        self.muscles: List[str] = kwargs.pop('muscles')
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['links'] = kwargs.pop('sens_links', [])
        options['joints'] = kwargs.pop('sens_joints', [])
        options['contacts'] = kwargs.pop('sens_contacts', [])
        options['xfrc'] = kwargs.pop('sens_xfrc', [])
        options['muscles'] = kwargs.pop('sens_muscles', [])
        return options

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))


class ModelOptions(Options):
    """Model options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
    ):
        super().__init__()
        self.sdf: str = sdf
        self.spawn: SpawnOptions = (
            spawn
            if isinstance(spawn, SpawnOptions)
            else SpawnOptions(**spawn)
        )


class AnimatOptions(ModelOptions):
    """Animat options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
            morphology: Union[MorphologyOptions, Dict],
            control: Union[ControlOptions, Dict],
    ):
        super().__init__(
            sdf=sdf,
            spawn=spawn,
        )
        self.morphology: MorphologyOptions = (
            morphology
            if isinstance(morphology, MorphologyOptions)
            else MorphologyOptions(**morphology)
        )
        self.control: ControlOptions = (
            control
            if isinstance(control, ControlOptions)
            else ControlOptions(**control)
        )


class WaterOptions(Options):
    """Water options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.sdf: str = kwargs.pop('sdf')
        self.drag: bool = kwargs.pop('drag')
        self.buoyancy: bool = kwargs.pop('buoyancy')
        self.height: float = kwargs.pop('height')
        self.velocity: List[float] = [*kwargs.pop('velocity')]
        self.viscosity: float = kwargs.pop('viscosity')
        self.density: float = kwargs.pop('density')
        self.maps: List = kwargs.pop('maps')


class ArenaOptions(ModelOptions):
    """Arena options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
            water: Union[WaterOptions, Dict],
            ground_height: float,
    ):
        super().__init__(sdf=sdf, spawn=spawn)
        self.water: WaterOptions = (
            water
            if isinstance(water, WaterOptions)
            else WaterOptions(**water)
        )
        self.ground_height = ground_height


class SiteOptions(Options):
    """ Add reference markers for motion tracking

    Parameters
    ----------
    name : str
        The name identifier for the object.
    shape : str [sphere, capsule, ellipsoid, cylinder, box], “sphere”
        The geometric shape type of the object.
    size: list[float] [0.005 0.005 0.005]
        Sizes of the geometric shape representing the site.
    pos : list[float]
        The 3D position coordinates [x, y, z] of the object.
    quat : list[float]
        The orientation quaternion [w, x, y, z] or [x, y, z, w]
        representing the object's rotation.
    rgba : list[float] or None, optional
        The color and opacity values [r, g, b, a] in range [0, 1].
        If None, a default color will be used. Default is None.
    """

    def __init__(
            self,
            name: str,
            shape: str,
            size: list[float],
            pos: list[float],
            quat: list[float],
            rgba: Union[list[float], None] = None,
    ):
        super().__init__()
        self.name = name
        self.shape = shape
        self.size = size
        self.pos = pos
        self.quat = quat
        if rgba is None:
            self.rgba = [1.0, 0.0, 0.0, 1.0]
        else:
            assert len(rgba) == 4
            self.rgba = rgba


# TRANSMISSION OPTIONS
# Not using StrEnum until Python 3.10 EOL
class TendonType(str, Enum):
    """Refer to MuJoCo docs for information about the tendon description and convention
    https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model"""
    FIXED = 'fixed'
    SPATIAL = 'spatial'


class TendonOptions(Options):
    """ Transmission Options """

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.type: TendonType = kwargs.pop('type')


class FixedTendonJointOptions(Options):
    """ Fixed Tendon Joint Options """

    def __init__(self, joint: str, coeff: float):
        self.joint = joint
        self.coeff = coeff


class FixedTendonOptions(TendonOptions):
    """ Fixed tendon that acts on a joints  """

    def __init__(self, **kwargs):
        name = kwargs.pop('name')
        super().__init__(name=name, type=TendonType.FIXED.value)
        # Each entry: {'name': 'joint1', 'coeff': 1.0}
        self.joints: List[FixedTendonJointOptions] = kwargs.pop('joints')


class SpatialTendonPathOptions(Options):
    """ Spatial tendon path options """

    def __init__(self, link: str, pos: List[float]):
        super().__init__()
        self.link = link
        self.pos = pos


class SpatialTendonOptions(TendonOptions):
    """ Spatial Tendons """

    def __init__(
            self,
            name: str,
            path: List[SpatialTendonPathOptions],
            len_range: List[float] = None
    ):
        super().__init__(name=name, type=TendonType.SPATIAL.value)
        # Each entry: {'link': 'femur', 'pos': [0.01, 0.02, 0.03]}
        self.path = path
        # Required if using MuJoCo Hill model
        self.len_range = len_range
        # For next iteration
        # self.geoms: List[str] = None   # Wrapping object


# Not using StrEnum until Python 3.10 EOL
class MuscleFrcDynTypes(str, Enum):
    """ Different Muscle Model Types """

    EKEBERG = 'ekeberg'
    HILL = 'hill'
    MUJOCO = 'mujoco'
    BROWN = 'brown'
    RIGIDTENDON = 'rigidtendon'


class MuscleFrcDynOptions(Options):
    """ Muscle Dynamics Options """

    def __init__(self, model: MuscleFrcDynTypes):
        self.model = model


class MuscleActDynOptions(Options):
    """ Muscle activation dynamics """

    def __init__(
            self,
            act_tconst: float,
            deact_tconst: float,
            init_act: float = 0.0
    ):
        super().__init__()

        self.act_tconst: float = act_tconst
        self.deact_tconst: float = deact_tconst
        self.init_act = init_act

    @classmethod
    def defaults(cls):
        """ Defaults """
        act_tconst = 0.01       # 10ms
        deact_tconst = 0.04     # 40ms
        return cls(act_tconst=act_tconst, deact_tconst=deact_tconst)


class MuscleSensorDynOptions(Options):
    """ Muscle sensor dynamics options """

    def __init__(self, model: MuscleFrcDynTypes):
        self.model = model


class EkebergFrcDynOptions(Options):
    """ Ekeberg muscle force options """

    def __init__(
        self,
        gain: float,
        stiffness: float,
        tonic_stiffness: float,
        damping: float
    ):
        self.gain = gain
        self.stiffness = stiffness
        self.tonic_stiffness = tonic_stiffness
        self.damping = damping


class HillDynOptions(MuscleFrcDynOptions):
    """ Hill Muscle Model Options """

    def __init__(
        self,
        max_force: float,
        optimal_fiber: float,
        tendon_slack: float,
        max_velocity: float,
        pennation_angle: float,
        act_dynamics: MuscleActDynOptions
    ):
        super().__init__(model=MuscleFrcDynTypes.HILL.value)
        self.max_force = max_force
        self.optimal_fiber = optimal_fiber
        self.tendon_slack = tendon_slack
        self.max_velocity = max_velocity
        self.pennation_angle = pennation_angle
        self.act_dynamics = act_dynamics


# Muscle sensory dynamics
class MuscleIaSensorOptions(Options):
    """ Ia Muscle sensor options """

    def __init__(
        self,
        kv: float,
        pv: float,
        k_dI: float,
        k_nI: float,
        const_I: float,
        l_ce_th: float,
    ):
        self.kv = kv
        self.pv = pv
        self.k_dI = k_dI
        self.k_nI = k_nI
        self.const_I = const_I
        self.l_ce_th = l_ce_th

    @classmethod
    def from_defaults(cls):
        return cls(
            kv=6.2/6.2,
            pv=0.6,
            k_dI=2.0/6.2,
            k_nI=0.06,
            const_I=0.05,
            l_ce_th=0.85,
        )


class MuscleIbSensorOptions(Options):
    """ Muscle Ib Sensors Options"""

    def __init__(self, kF: float):
        self.kF = kF

    @classmethod
    def from_defaults(cls):
        return cls(kF=1.0)


class MuscleIISensorOptions(Options):
    """ Muscle II Sensors Options"""

    def __init__(
        self,
        k_dII,
        k_nII,
        const_II,
        l_ce_th,
    ):
        self.k_dII = k_dII
        self.k_nII = k_nII
        self.const_II = const_II
        self.l_ce_th = l_ce_th

    @classmethod
    def from_defaults(cls):
        return cls(
            k_dII=1.5,
            k_nII=0.06,
            const_II=0.05,
            l_ce_th=0.85,
        )


class MuscleOptions(Options):
    """ Muscle Options """

    def __init__(
        self,
        name: str,
        tendon: Union[SpatialTendonOptions, FixedTendonOptions],
        frc_dynamics: MuscleFrcDynOptions,
        act_dynamics: MuscleActDynOptions,
        sensor_dynamics: MuscleSensorDynOptions
    ):
        super().__init__()
        self.name = name
        # Tendon
        self.tendon = tendon
        # Muscle dynamics
        self.frc_dynamics = frc_dynamics
        # Activation Dynamics
        self.act_dynamics = act_dynamics
        # Sensor dynamics
        self.sensor_dynamics = sensor_dynamics

        # self.lmin: float = kwargs.pop(
        #     'lmin',
        #     self.lmtu_min-self.tendon_slack/self.optimal_fiber
        # )
        # self.lmax: float = kwargs.pop(
        #     'lmax',
        #     self.lmtu_max-self.tendon_slack/self.optimal_fiber
        # )
        # # initialization
        # self.init_fiber: float = kwargs.pop('init_fiber', self.optimal_fiber)
