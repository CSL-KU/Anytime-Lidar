from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .centerpoint_opt import CenterPointOpt
from .centerpoint_valo import CenterPointVALO
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet_opt import PillarNetOpt
from .pillarnet_valo import PillarNetVALO
from .mural import MURAL
from .voxelnext import VoxelNeXt
from .voxelnext_anytime import VoxelNeXtAnytime
from .transfusion import TransFusion
from .transfusion_anytime import TransFusionAnytime
from .dsvt_centerhead_opt import DSVT_CenterHead_Opt
from .dsvt_centerhead_valo import DSVT_CenterHead_VALO

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'CenterPointOpt': CenterPointOpt,
    'CenterPointVALO': CenterPointVALO,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'PillarNetOpt': PillarNetOpt,
    'PillarNetVALO': PillarNetVALO,
    'MURAL': MURAL,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'VoxelNeXt': VoxelNeXt,
    'VoxelNeXtAnytime': VoxelNeXtAnytime,
    'TransFusion': TransFusion,
    'TransFusionAnytime': TransFusionAnytime,
    'DSVT_CenterHead_Opt': DSVT_CenterHead_Opt,
    'DSVT_CenterHead_VALO': DSVT_CenterHead_VALO,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
