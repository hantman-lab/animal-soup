import pandas as pd
from ipywidgets import HBox, VBox, Select, Button, Layout, RadioButtons
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector, LinearRegionSelector
import numpy as np
from ._ethogram import EthogramVizContainer, ETHOGRAM_COLORS
from pathlib import Path
from typing import Union
from .batch_utils import validate_path

class EthogramCleaner(EthogramVizContainer):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
            clean_df: Union[str, Path] = None
    ):
        super(EthogramCleaner, self).__init__(
            dataframe=dataframe,
            start_index=start_index
        )

        if clean_df is None:
            df_dir, relative_path = self._dataframe.paths.split(self._dataframe.paths.get_df_path())
            self.clean_df = self._dataframe.copy(deep=True)
            # set ethograms to empty dicts() in order to clean ethograms
            self.clean_df["ethograms"] = [dict() for i in range(len(self._dataframe.index))]
            # save clean df to disk in same place as current dataframe
            self.clean_df.to_hdf(df_dir.with_name(f'{relative_path.stem}_cleaned').with_suffix('.hdf'), key='df')
        elif isinstance(clean_df, str):
            path = Path(clean_df)
            validate_path(path)
            self.clean_df = pd.read_hdf(path)
        else:
            validate_path(clean_df)
            self.clean_df = pd.read_hdf(clean_df)

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row]

        if self.plot is None:
            self.plot = Plot(size=(500, 100))

        self.ethogram_array = row["ethograms"][self.selected_trial]

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=10,
                name=b
            )

            lg.colors = 0
            lg.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -10) - 1
            lg.position_y = y_pos

        self.ethogram_selector = LinearRegionSelector(
            bounds=(0, 50),
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            origin=(0, -25),
            fill_color=(0, 0, 0, 0),
            parent=lg,
            size=(55),
        )

        self.plot.add_graphic(self.ethogram_selector)
        self.ethogram_selector.edges[0].add_event_handler(self.ethogram_event_handler_left, "pointer_enter")
        self.ethogram_selector.edges[1].add_event_handler(self.ethogram_event_handler_right, "pointer_enter")
        self.plot.auto_scale()

    def ethogram_event_handler_left(self, ev):
        """
        Event handler called for linear region selector left bound.
        """
        ix = self.ethogram_selector.get_selected_indices()[0]
        self.image_widget.sliders["t"].value = ix

    def ethogram_event_handler_right(self, ev):
        """
        Event handler called for linear region selector right bound.
        """
        ix = self.ethogram_selector.get_selected_indices()[-1]
        self.image_widget.sliders["t"].value = ix

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                self.trial_selector
        ]),
            self.plot.show()])