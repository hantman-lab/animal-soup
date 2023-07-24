"""
This software license is the 3-clause BSD license plus a fourth clause that
prohibits redistribution for commercial purposes without further permission.

BSD 3-Clause License

Copyright (c) 2022, Kushal Kolar.
"""

from typing import *
from pathlib import Path
from warnings import warn

import numpy as np
from decord import VideoReader

slice_or_int_or_range = Union[int, slice, range]


class LazyVideo:
    def __init__(
        self,
        path: Union[Path, str],
        min_max: Tuple[int, int] = None,
        as_grayscale: bool = False,
        rgb_weights: Tuple[float, float, float] = (0.299, 0.587, 0.114),
        **kwargs,
    ):
        """
        LazyVideo reader, basically just a wrapper for ``decord.VideoReader``.
        Should support opening anything that decord can open.

        **Important:** requires ``decord`` to be installed: https://github.com/dmlc/decord

        Parameters
        ----------
        path: Path or str
            path to video file

        min_max: Tuple[int, int], optional
            min and max vals of the entire video, uses min and max of 10th frame if not provided

        as_grayscale: bool, optional
            return grayscale frames upon slicing

        rgb_weights: Tuple[float, float, float], optional
            (r, g, b) weights used for grayscale conversion if ``as_graycale`` is ``True``.
            default is (0.299, 0.587, 0.114)

        kwargs
            passed to ``decord.VideoReader``

        """
        self._video_reader = VideoReader(str(path), **kwargs)

        try:
            frame0 = self._video_reader[10].asnumpy()
        except IndexError:
            frame0 = self._video_reader[0].asnumpy()

        self._shape = (self._video_reader._num_frame, *frame0.shape[:-1])

        if len(frame0.shape) > 2:
            # we assume the shape of a frame is [x, y, RGB]
            self._is_color = True
        else:
            # we assume is already grayscale
            self._is_color = False

        self._dtype = frame0.dtype

        if min_max is not None:
            self._min, self._max = min_max
        else:
            self._min = frame0.min()
            self._max = frame0.max()

        self.as_grayscale = as_grayscale
        self.rgb_weights = rgb_weights

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        """[n_frames, x, y], RGB color dim not included in shape"""
        return self._shape

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self._min

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self._max

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int
            number of bytes for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,))

    @property
    def nbytes_gb(self) -> float:
        """
        float
            number of gigabytes for the array if it were fully computed
        """
        return self.nbytes / 1e9

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        if not self.as_grayscale:
            return self._video_reader[indices].asnumpy()

        if self._is_color:
            a = self._video_reader[indices].asnumpy()

            # R + G + B -> grayscale
            gray = (
                a[..., 0] * self.rgb_weights[0]
                + a[..., 1] * self.rgb_weights[1]
                + a[..., 2] * self.rgb_weights[2]
            )

            return gray

        warn("Video is already grayscale, just returning")
        return self._video_reader[indices].asnumpy()

    def as_numpy(self):
        """
        NOT RECOMMENDED, THIS COULD BE EXTREMELY LARGE. Converts to a standard numpy array in RAM.

        Returns
        -------
        np.ndarray
        """
        warn(
            f"\nYou are trying to create a numpy.ndarray from a LazyArray, "
            f"this is not recommended and could take a while.\n\n"
            f"Estimated size of final numpy array: "
            f"{self.nbytes_gb:.2f} GB"
        )
        a = np.zeros(shape=self.shape, dtype=self.dtype)

        for i in range(self.n_frames):
            a[i] = self[i]

        return a

    def __getitem__(self, item: Union[int, Tuple[slice_or_int_or_range]]):
        if isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
            indexer = item

        elif isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            indexer = item[0]

        else:
            raise IndexError(
                f"You can index LazyArrays only using slice, int, or tuple of slice and int, "
                f"you have passed a: <{type(item)}>"
            )

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if start is not None:
                if start > self.n_frames:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame start index of <{start}> "
                        f"lies beyond `n_frames` <{self.n_frames}>"
                    )
            if stop is not None:
                if stop > self.n_frames:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame stop index of <{stop}> "
                        f"lies beyond `n_frames` <{self.n_frames}>"
                    )

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = slice(start, stop, step)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, int):
            return self._compute_at_indices(indexer)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} @{hex(id(self))}\n"
            f"{self.__class__.__doc__}\n"
            f"Frames are computed only upon indexing\n"
            f"shape [frames, x, y]: {self.shape}\n"
        )
