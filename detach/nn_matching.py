# vim: expandtab:ts=4:sw=4
import numpy as np
from scipy.ndimage import convolve
from datetime import timedelta
from monai.metrics import compute_average_surface_distance
from .track import generate_coordinate_feature
from . import linear_assignment

def _nn_surface_distance(x, y, reduction='min', batch_size=-1, decay_paras=None):
    """ Helper function for nearest neighbor distance metric (surface).

    Parameters
    ----------
    x : list of ndarray
        A list of N 2d-image-array (sample images).
    y : list of ndarray
        A list of M 2d-image-array (query points).
    reduction : str
        reduction method.
        accept values: "none", "min", "max", "sum", "decay"
    batch_size : Ignored
        Not used, present here for API consistency by convention
    decay_paras : 
        Dict{"lambda" -> float, "A" -> float,
            "t_x" -> list[datetime], "t_y" -> list[datetime]}
        parameters for decay reduction (reduction="decay")
        formula for decay reduction:
            weight = exp(-lamda*dt)*(1+A*cos(2*pi*dt/T))
            where dt is the delta time, T is time period (27 days).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        average surface distance to a sample in `x`.

    """
    q_x = np.array(x).reshape(-1, 1, 1, 180, 360).repeat(len(y), axis=1).reshape((-1, 1, 180, 360))
    q_y = np.array(y).reshape(1, -1, 1, 180, 360).repeat(len(x), axis=0).reshape((-1, 1, 180, 360))
    dis_array = compute_average_surface_distance(q_x, q_y, symmetric=True).numpy().reshape(len(x), len(y))
    if reduction == 'none':
        return dis_array
    elif reduction == 'sum':
        return np.sum(dis_array, axis=0)
    elif reduction == 'max':
        return np.max(dis_array, axis=0)
    elif reduction == 'min':
        return np.min(dis_array, axis=0)
    elif reduction == "decay":
        decay_keys = ['decay_lambda', 'decay_A', 't_x', 't_y']
        if decay_paras is None or any(k not in decay_paras for k in decay_keys):
            print("lack of decay parameters, use minimum reduction instead")
            return np.min(dis_array, axis=0)
        decay_lambda = decay_paras['decay_lambda']
        decay_A = decay_paras['decay_A']
        t_yy, t_xx = np.meshgrid(decay_paras['t_y'], decay_paras['t_x'])
        delta_t = (np.abs((t_xx - t_yy)/timedelta(days=27))).astype(np.float_)
        weight = np.exp(-decay_lambda*delta_t*27)*(1+decay_A*np.cos(2*np.pi*delta_t))
        return np.average(dis_array, axis=0, weights=weight)
    return np.min(dis_array, axis=0)


def _discrete_haversine_distance(pos_1_list, pos_2_list):
    # convert deg to rad
    lon1 = pos_1_list[0]*np.pi/180
    lat1 = pos_1_list[1]*np.pi/180
    lon2 = pos_2_list[0]*np.pi/180
    lat2 = pos_2_list[1]*np.pi/180
    # calculate distance on sphere with haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    delta_rad = 2*np.arcsin(np.sqrt(a))
    return 180*delta_rad/np.pi

def _collection_haversine_distance(pos_1_list, pos_2_list):
    length_1 = pos_1_list.shape[1]
    length_2 = pos_2_list.shape[1]
    pos_1_list = pos_1_list.reshape(2, -1, 1).repeat(length_2, axis=2)
    pos_2_list = pos_2_list.reshape(2, 1, -1).repeat(length_1, axis=1)
    return _discrete_haversine_distance(pos_1_list, pos_2_list)

def _nn_spherical_surface_distance(x, y, reduction='min', batch_size=-1, decay_paras=None):
    """ Helper function for surface distance on sphere.

    Parameters
    ----------
    x : list of ndarray
        A list of N 2d-image-array (sample images).
    y : list of ndarray
        A list of M 2d-image-array (query points).
    reduction : str
        reduction method.
        accept values: "none", "min", "max", "sum", "decay"
    batch_size : int
        batch size for calculating distance
        if batch_size <= 0, means no batch size
    decay_paras : 
        Dict{"lambda" -> float, "A" -> float,
            "t_x" -> list[datetime], "t_y" -> list[datetime]}
        parameters for decay reduction (reduction="decay")
        formula for decay reduction:
            weight = exp(-lamda*dt)*(1+A*cos(2*pi*dt/T))
            where dt is the delta time, T is time period (27 days).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        average surface distance to a sample in `x`.

    """
    q_x = np.array(x).reshape(-1, 180, 360)
    q_y = np.array(y).reshape(-1, 180, 360)
    kernel = np.array([[0, 1, 0], [1, 5, 1], [0, 1, 0]])
    neighbor_codemap_x = convolve(q_x.astype(np.uint8), kernel, mode='wrap', axes=(1, 2))
    border_x = ((neighbor_codemap_x>4)&(neighbor_codemap_x<9))
    x_img_id, x_lat, x_lon = np.nonzero(border_x)
    x_lat = 90 - x_lat
    neighbor_codemap_y = convolve(q_y.astype(np.uint8), kernel, mode='wrap', axes=(1, 2))
    border_y = ((neighbor_codemap_y>4)&(neighbor_codemap_y<9))
    y_img_id, y_lat, y_lon = np.nonzero(border_y)
    y_lat = 90 - y_lat

    area_x = np.sum(q_x, axis=(1, 2))
    area_y = np.sum(q_y, axis=(1, 2))
    area_yy, area_xx = np.meshgrid(area_y, area_x)
    dis_array_1 = np.zeros(shape=(len(x), len(y)))
    dis_array_2 = np.zeros(shape=(len(x), len(y)))

    if batch_size <= 0:
        batch_size = max(len(x), len(y))
    for i in range(0, len(x), batch_size):
        for j in range(0, len(y), batch_size):
            i_end = min(i+batch_size, len(x))
            j_end = min(j+batch_size, len(y))
            i_mask = (x_img_id>=i)&(x_img_id<i_end)
            j_mask = (y_img_id>=j)&(y_img_id<j_end)
            dis_matrix = _collection_haversine_distance(np.array([x_lon, x_lat])[:, i_mask],
                                                        np.array([y_lon, y_lat])[:, j_mask])
            for ii in range(i, i_end):
                x_mask = (x_img_id[i_mask]==ii)
                for jj in range(j, j_end):
                    y_mask = (y_img_id[j_mask]==jj)
                    if np.all(~x_mask) or np.all(~y_mask):
                        dis_array_1[ii, jj] = np.inf
                        dis_array_2[ii, jj] = np.inf
                    else:
                        dis_array_1[ii, jj] = np.mean(np.min(dis_matrix[x_mask][:, y_mask],axis=1))
                        dis_array_2[ii, jj] = np.mean(np.min(dis_matrix[x_mask][:, y_mask],axis=0))
    dis_array = (dis_array_1*area_xx+dis_array_2*area_yy)/(area_xx+area_yy)
    if reduction == 'none':
        return dis_array
    elif reduction == 'sum':
        return np.sum(dis_array, axis=0)
    elif reduction == 'max':
        return np.max(dis_array, axis=0)
    elif reduction == 'min':
        return np.min(dis_array, axis=0)
    elif reduction == "decay":
        decay_keys = ['decay_lambda', 'decay_A', 't_x', 't_y']
        if decay_paras is None or any(k not in decay_paras for k in decay_keys):
            print("lack of decay parameters, use minimum reduction instead")
            return np.min(dis_array, axis=0)
        decay_lambda = decay_paras['decay_lambda']
        decay_A = decay_paras['decay_A']
        t_yy, t_xx = np.meshgrid(decay_paras['t_y'], decay_paras['t_x'])
        delta_t = (np.abs((t_xx - t_yy)/timedelta(days=27))).astype(np.float_)
        weight = np.exp(-decay_lambda*delta_t*27)*(1+decay_A*np.cos(2*np.pi*delta_t))
        return np.average(dis_array, axis=0, weights=weight)
    return np.min(dis_array, axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        "surface" or "spherical_surface".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    reduction : str or dict
        reduction method.
        accept str values: "none", "min", "max", "sum", "decay"
        accept dict values: {"method" -> str, "decay_lambda" -> float, "decay_A" -> float}
    batch_size : int
        batch size for calculating distance
        if batch_size <= 0, means no batch size

    Attributes
    ----------
    samples : Dict[int -> List[Dict{'cls' -> int, 'hgc' -> ndarray, 'time' -> datetime}]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None,
                 reduction="min", batch_size=-1):
        if metric == "surface":
            self._metric = _nn_surface_distance
        elif metric == "spherical_surface":
            self._metric = _nn_spherical_surface_distance
        else:
            raise ValueError(
                "Invalid metric; must be 'surface' or 'spherical_surface'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        if type(reduction) == str:
            self.reduction = {"method": reduction}
        else:
            self.reduction = reduction
        self.batch_size = batch_size
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : list
            A list of the class and hgc binaray mask.
            (Dict{'cls'->int, 'hgc'->ndarray, 'time'->datetime})
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, detections, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        detection : list
            A list of the database.Instance object
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        features = [generate_coordinate_feature(det) for det in detections]
        features_cls = np.array([feat['cls'] for feat in features])
        features_hgc = np.array([feat['hgc'] for feat in features])
        features_time = np.array([feat['time'] for feat in features])
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            target_list = [t for t in self.samples[target]]
            target_cls_list = [t['cls'] for t in target_list]
            target_hgc = np.array([t['hgc'] for t in target_list])
            if len(set(target_cls_list)) != 1:
                print(f"possible wrong class for track: {len(set(target_cls_list))}")
            feature_id_mask = (features_cls==target_cls_list[0])
            decay_paras = None
            if self.reduction['method'] == 'decay':
                target_time = np.array([t['time'] for t in target_list])
                decay_paras = {'decay_lambda': self.reduction['decay_lambda'],
                               'decay_A': self.reduction['decay_A'],
                               't_x': target_time, 't_y': features_time[feature_id_mask]}
            cost_matrix[i, feature_id_mask] = self._metric(target_hgc, features_hgc[feature_id_mask],
                                                           reduction=self.reduction['method'],
                                                           batch_size=self.batch_size,
                                                           decay_paras=decay_paras)
            cost_matrix[i, ~feature_id_mask] = linear_assignment.INFTY_COST
        return cost_matrix
