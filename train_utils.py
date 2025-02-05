from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
import torch 
import math

def neg_par_log_likelihood(pred,ytime,yevent):
    n_observed = yevent.sum(0)
    if(n_observed==0):
        return 0.0
    
    ytime_indicator = R_set(ytime).float()
    
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    # pred = torch.tensor(pred.values).cuda()
    
    risk_set_sum = ytime_indicator.mm(torch.exp(pred.float()))
    
    diff = pred - torch.log(risk_set_sum)
    
    yevent = yevent[:,None]
    
    sum_diff_in_observed = torch.transpose(diff,0,1).mm(yevent.float())
    
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    
    return cost
#https://github.com/tomcat123a/survival_loss_criteria/blob/master/loss.py
def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
  
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.triu(matrix_ones)
    

	return(indicator_matrix)
    
def CoxLoss(estimate, event, time):
    """
    estimate : (B, 1)
    event : (B, 1)
    time : (B, 1)
    B : Batch size
    """
    loss = cox.neg_partial_log_likelihood(estimate, event.bool(), time)
    return loss

def Cindex(estimate, event, time):
    cindex = ConcordanceIndex()
    c = cindex(estimate, event.bool(), time)
    return c

class MTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.
    
    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time interval.
    
    Note that a slightly more efficient reformulation is used here, first proposed
    in [2]_.
    
    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival distributions 
    as a sequence of dependent regressors’, in Advances in neural information processing systems 24,
    2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn Consumer-Specific Reservation Price Distributions’,
    Master's thesis, University of Alberta, Edmonton, AB, 2015.
    """
    def __init__(self, in_features, num_time_bins):
        """Initialises the module.
        
        Parameters
        ----------
        in_features : int
            Number of input features.
        num_time_bins : int
            The number of bins to divide the time axis into.
        """
        super().__init__()
        self.in_features = in_features
        self.num_time_bins = num_time_bins

        weight = torch.zeros(self.in_features,
                             self.num_time_bins-1,
                             dtype=torch.float)
        bias = torch.zeros(self.num_time_bins-1)
        self.mtlr_weight = nn.Parameter(weight)
        self.mtlr_bias = nn.Parameter(bias)
        
        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer("G", 
                             torch.tril(torch.ones(self.num_time_bins-1, 
                                                   self.num_time_bins, requires_grad=True)))
        self.reset_parameters()

    def forward(self, x):
        """Performs a forward pass on a batch of examples.
        
        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.
        
        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)
    
    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)


def masked_logsumexp(x, mask, dim=-1):
    """Computes logsumexp over elements of a tensor specified by a mask in a numerically stable way.
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    mask : torch.Tensor
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim : int
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.
    
    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask, dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits, target, average=False):
    """Computes the negative log-likelihood of a batch of model predictions.
    
    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one instance
        in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
        
    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used during training.
    """
    censored = target.sum(dim=1) > 1
 
    if censored.any():
        nll_censored = masked_logsumexp(logits[censored], target[censored]).sum()
    else:
        nll_censored = 0.
    if (~censored).any():
        nll_uncensored = (logits[~censored] * target[~censored]).sum()
    else:
        nll_uncensored = 0.
    
    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()
    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)
    
    return nll_total


def mtlr_survival(logits):
    """Generates predicted survival curves from predicted logits.
    
    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one instance
        in each row.
        
    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used during training.
    """
    # TODO: do not reallocate G in every call
    G = torch.tril(torch.ones(logits.size(1), logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)


def mtlr_survival_at_times(logits, train_times, pred_times):
    """Generates predicted survival curves at arbitrary timepoints using linear interpolation.
    
    This function uses scipy.interpolate internally and returns a Numpy array, in contrast
    with `mtlr_survival`.
    
    Parameters
    ----------
    logits : torch.Tensor 
        Tensor with the time-logits (as returned by the MTLR module) for one instance
        in each row.
    train_times : Tensor or ndarray
        Time bins used for model training. Must have the same length as the first dimension
        of `pred`.
    pred_times : np.ndarray
        Array of times used to compute the survival curve.
    
    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values are linearly interpolated
        at timepoints not used for training.
    """
    with torch.no_grad():
        surv = mtlr_survival(logits).cpu().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(pred_times)


def mtlr_hazard(logits):
    """Computes the hazard function from MTLR predictions.
    
    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t, respectively.
    
    Parameters
    ----------
    logits : torch.Tensor
        The predicted logits as returned by the `MTLR` module.
        
    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits):
    """Computes the overall risk of event from MTLR predictions.
    
    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.
    
    Parameters
    ----------
    logits : torch.Tensor
        The predicted logits as returned by the `MTLR` module.
        
    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)

def make_time_bins(times, num_bins=None, use_quantiles=True):
    """Creates the bins for survival time discretisation.
    
    By default, sqrt(num_observation) bins corresponding to the quantiles of 
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.
    
    Parameters
    ----------
    times : np.ndarray
        Array of survival times.
    num_bins : int, optional
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles : bool
        If True, the bin edges will correspond to quantiles of `times` (default).
        Otherwise, generates equally-spaced bins.
        
    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    return bins

# utility functions
def encode_survival(time, event, bins):
    """Encodes survival time and event indicator in the format
    required for MTLR training.
    
    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and 
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.
    
    Parameters
    ----------
    time : np.ndarray
        Array of event or censoring times.
    event : np.ndarray
        Array of event indicators (0 = censored).
    bins : np.ndarray
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    time = np.clip(time, 0, bins.max())
    bin_idxs = np.digitize(time, bins)
    # add extra bin [max_time, inf) at the end
    y = np.zeros((time.shape[0], bins.shape[0] + 1))#
    for i, e in enumerate(event):
        bin_idx = bin_idxs[i]
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return torch.tensor(y, dtype=torch.float)

def TCGA_csv_preprocess(df, cols_required, cols_to_one_hot,time_col, event_col):
    '''
    df : pd.DataFrame
    cols_required : List
    cols_to_one_hot : List
    '''
    df = df[cols_required]
    df = df.rename(columns={time_col :"time"})
    df = df.rename(columns={event_col :"event"})
    
    for col in cols_to_one_hot:
        one_hot_encoded = pd.get_dummies(df[col])
        df = pd.concat([df,one_hot_encoded],axis=1)
    # cols_to_one_hot.remove('histological_type')
    df = df.drop(cols_to_one_hot, axis = 1)
    df.dropna(inplace = True)
    return df

# Define DataLoader for the SurvivalDataset
def create_dataloader(feat_paths, surv_data, clinical_data, transform=None, batch_size=32, num_patches = 10000):
    dataset = SurvivalDataset(feat_paths, surv_data, clinical_data, num_patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
