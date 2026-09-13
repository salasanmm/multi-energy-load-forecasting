import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable


def mape_loss(target, input):
    return (torch.abs(input - target) / (torch.abs(target) + 1e-2)).mean() * 100


def MAPE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))

    # e = (y_true + y_pre) / 2 + 1e-2
    # re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    re = np.mean(np.abs((y_true - y_pre) / y_true)) * 100

    return re


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderS(object):
    """Chronological loader for hourly multi-energy data.

    ``train`` and ``valid`` are fractions of the complete timeline; the test
    fraction is ``1 - train - valid``.  In addition to the tensors used by the
    model, the loader keeps calendar metadata for each forecast endpoint.  The
    source CSV does not contain a timestamp column, but it does contain the
    ``DayOfYear_cos`` feature.  We decode that feature (including missing-day
    gaps) so that seasonal test-set metrics can be reported without changing
    the model input format.
    """
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2,
                 exclude_columns=None):
        self.P = window
        self.h = horizon
        if train <= 0 or valid < 0 or train + valid >= 1:
            raise ValueError("train and valid fractions must satisfy 0 < train, 0 <= valid, train + valid < 1")

        with open(file_name, encoding='utf-8-sig') as fin:
            full_header = [item.strip() for item in fin.readline().strip().split(',')]
        fin = open(file_name)
        full_rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
        fin.close()
        excluded = {str(name).strip().lower() for name in (exclude_columns or [])}
        keep_indices = [i for i, name in enumerate(full_header)
                        if name.strip().lower() not in excluded]
        if len(keep_indices) < 3 or keep_indices[:3] != [0, 1, 2]:
            raise ValueError("The first three columns (KW, CHWTON, HTmmBTU) must remain model targets")
        self.input_columns = [full_header[i] for i in keep_indices]
        self.excluded_columns = [name for i, name in enumerate(full_header) if i not in keep_indices]
        self.rawdat = full_rawdat[:, keep_indices]
        header = self.input_columns
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.train_end = int(train * self.n)
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        valid_end = int((train + valid) * self.n)
        self.train_feas = self.dat[:self.train_end, :]
        self._infer_calendar(header)
        self._split(self.train_end, valid_end, self.n)

        self.scale = torch.from_numpy(self.scale[:3]).float()

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.device = device

    def _infer_calendar(self, header):
        """Recover day-of-year labels from the encoded calendar feature.

        ``DayOfYear_cos`` in this dataset is ``(cos(2*pi*d/365)+1)/2``.  The
        cosine is symmetric, so each value has two possible day numbers.  We
        resolve the ambiguity by following the chronological sequence and
        allowing forward jumps when source days are missing.  If the feature is
        absent, a documented hourly fallback is used.
        """
        try:
            doy_col = next(i for i, name in enumerate(header)
                           if name.lower().replace('_', '') == 'dayofyearcos')
        except StopIteration:
            doy_col = None

        if doy_col is None:
            absolute_day = np.arange(self.n, dtype=np.int64) // 24
            self.row_day_of_year = (absolute_day % 365) + 1
            self.row_absolute_day = absolute_day
            self.calendar_source = 'hourly_index_fallback'
            return

        encoded = np.asarray(self.rawdat[:, doy_col], dtype=float)
        if encoded.size == 0 or not np.isfinite(encoded).all():
            absolute_day = np.arange(self.n, dtype=np.int64) // 24
            self.row_day_of_year = (absolute_day % 365) + 1
            self.row_absolute_day = absolute_day
            self.calendar_source = 'hourly_index_fallback'
            return

        # Identify contiguous rows belonging to the same calendar day.
        starts = np.r_[0, np.flatnonzero(np.abs(np.diff(encoded)) > 1e-10) + 1]
        run_values = encoded[starts]
        run_lengths = np.diff(np.r_[starts, self.n])

        day_numbers = np.arange(1, 366, dtype=float)
        encoded_grid = (np.cos(2.0 * np.pi * day_numbers / 365.0) + 1.0) / 2.0
        inferred_doy = []
        previous = None
        for run_value in run_values:
            distances = np.abs(encoded_grid - run_value)
            # The cosine has at most two equally good candidates.
            nearest = np.argsort(distances)[:2]
            candidates = [int(day_numbers[i]) for i in nearest]
            if previous is None:
                # The ASU file starts on January 1 (d=1).
                choice = min(candidates, key=lambda d: abs(d - 1))
            else:
                forward_distance = [(d - previous) % 365 for d in candidates]
                choice = candidates[int(np.argmin(forward_distance))]
            inferred_doy.append(choice)
            previous = choice

        absolute_runs = [inferred_doy[0] - 1]
        for previous_doy, current_doy in zip(inferred_doy[:-1], inferred_doy[1:]):
            absolute_runs.append(absolute_runs[-1] + (current_doy - previous_doy) % 365)

        self.row_day_of_year = np.repeat(np.asarray(inferred_doy, dtype=np.int64), run_lengths)
        self.row_absolute_day = np.repeat(np.asarray(absolute_runs, dtype=np.int64), run_lengths)
        self.calendar_source = 'DayOfYear_cos'

    @staticmethod
    def season_from_day_of_year(day_of_year):
        """Return four meteorological-season labels for one or more day numbers."""
        doy = np.asarray(day_of_year, dtype=np.int64)
        labels = np.full(doy.shape, 'spring', dtype=object)
        labels[(doy >= 152) & (doy <= 243)] = 'summer'       # Jun-Aug
        labels[(doy >= 244) & (doy <= 334)] = 'autumn'       # Sep-Nov
        labels[(doy >= 335) | (doy <= 59)] = 'winter'        # Dec-Feb
        return labels

    def get_split_day_of_year(self, split='test'):
        """Calendar day for each forecast endpoint in a split."""
        if split not in self.split_indices:
            raise ValueError(f"unknown split {split!r}; expected train, valid, or test")
        return self.row_day_of_year[self.split_indices[split]]

    def get_season_masks(self, split='test'):
        """Boolean masks grouping forecast endpoints by season."""
        labels = self.season_from_day_of_year(self.get_split_day_of_year(split))
        return {name: labels == name for name in ('spring', 'summer', 'autumn', 'winter')}

    def _normalized(self, normalize):

        for i in range(self.m):
            # Fit scaling parameters on training observations only.
            self.scale[i] = np.max(np.abs(self.rawdat[:self.train_end, i]))
            if self.scale[i] == 0:
                self.scale[i] = 1.0
            self.dat[:, i] = self.rawdat[:, i] / self.scale[i]


    def _split(self, train, valid, test):
        train_set = np.arange(self.P + self.h - 1, train, dtype=np.int64)
        valid_set = np.arange(train, valid, dtype=np.int64)
        test_set = np.arange(valid, self.n, dtype=np.int64)
        self.split_indices = {
            'train': train_set,
            'valid': valid_set,
            'test': test_set,
        }
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        # print("datshape", self.dat.shape)
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i] + 1 - horizon:idx_set[i] + 1, :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))
