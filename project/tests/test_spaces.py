import unittest

import numpy as np
import sklearn.linear_model
import xarray as xr

from project.spaces import Detrend


class TestDetrend(unittest.TestCase):
    def test_linear(self):
        x = np.arange(0, 10)[np.newaxis, :]
        coeffs = np.arange(1, 5)[:, np.newaxis]
        y = coeffs @ x
        da = xr.DataArray(
            y, coords=dict(state=np.arange(coeffs.shape[0]), time=x.squeeze())
        )
        # Convert to chunked DataArray
        da = da.to_dataset(name="var").chunk(chunks=dict(time=2))["var"]

        detrend = Detrend()
        detrend.fit(da)
        self.assertTrue(np.allclose(coeffs.squeeze(), detrend.coeffs))

        da_detrended = detrend.forward(da)
        self.assertTrue(np.allclose(da_detrended.data, 0))

    def test_random(self):
        x = np.random.randn(10, 100)
        da = xr.DataArray(
            x, coords=dict(state=np.arange(x.shape[0]), time=np.arange(x.shape[1]))
        )
        # Convert to chunked DataArray
        da = da.to_dataset(name="var").chunk(chunks=dict(time=2))["var"]

        sklearn_model = sklearn.linear_model.LinearRegression()
        sklearn_model.fit(da.time.data[:, np.newaxis], x.T)
        detrended_sklearn = x - sklearn_model.predict(da.time.data[:, np.newaxis]).T

        detrend = Detrend()
        detrend.fit(da)
        da_detrended = detrend.forward(da).compute()
        self.assertTrue(np.allclose(da_detrended.data, detrended_sklearn))

        da_retrended = detrend.backward(da_detrended).compute()
        self.assertTrue(np.allclose(da_retrended.data, da.data))
