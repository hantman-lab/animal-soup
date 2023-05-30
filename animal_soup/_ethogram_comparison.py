from ._behavior import BehaviorVizContainer, ETHOGRAM_COLORS
import pandas as pd
import numpy as np
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector, Synchronizer
from mesmerize_core.arrays import LazyVideo
from ipywidgets import HBox, VBox


class EthogramComparison(BehaviorVizContainer):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 start_index: int = 0,
                 ):
        super(EthogramComparison, self).__init__(
            dataframe=dataframe,
            start_index=start_index
        )

        self.deg_plot = None

        self._make_deg_ethogram_plot()

    def _make_deg_ethogram_plot(self, trial_index: int = 0):
        row = self._dataframe.iloc[self.current_row]
        session_dir = self.local_parent_path.joinpath(row['animal_id'], row['session_id'])

        if self.deg_plot is None:
            self.deg_plot = Plot(size=(500, 100))

        try:
            self.deg_ethogram_array = np.load(session_dir.joinpath(self.trial_selector.value).with_suffix('.npy'))
        except FileNotFoundError:
            self.deg_plot.clear()
            return

        y_bottom = 0
        for i, b in enumerate(self.behaviors):
            xs = np.arange(self.deg_ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg = self.deg_plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=10,
                name=b
            )

            lg.colors = 0
            lg.colors[self.deg_ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -10) - 1
            lg.position_y = y_pos

        self.deg_ethogram_selector = LinearSelector(
            selection=0,
            limits=(0, self.deg_ethogram_array.shape[1]),
            axis="x",
            parent=lg,
            end_points=(y_bottom, y_pos),
        )

        self.deg_plot.add_graphic(self.deg_ethogram_selector)

        self.sync = Synchronizer(self.deg_ethogram_selector, self.ethogram_selector, key_bind=None)

        self.deg_ethogram_selector.selection.add_event_handler(self.deg_ethogram_event_handler)
        self.deg_plot.camera.maintain_aspect = False
        self.deg_plot.auto_scale()

    def deg_ethogram_event_handler(self, ev):
        """
        Event handler called for linear selector.
        """
        ix = ev.pick_info["selected_index"]
        self.image_widget.sliders["t"].value = ix

    def _trial_change(self, obj):
        """
        Event handler called when a trial is changed in self.trial_selector.
        Updates the behavior imagewidget and ethogram plot with new data.
        """
        row = self._dataframe.iloc[self.current_row]

        session_path = self.local_parent_path.joinpath(row['animal_id'], row['session_id'])
        selected_video = session_path.joinpath(self.trial_selector.value).with_suffix('.avi')

        self.image_widget._data = [LazyVideo(selected_video)]
        self.image_widget.current_index["t"] = 0
        self.image_widget.sliders["t"].value = 0
        self.image_widget.plot.graphics[0].data = self.image_widget._data[0][0]

        trial_index = int(selected_video.stem.split('_v')[-1]) - 1
        self.plot.clear()
        self._make_ethogram_plot(trial_index=trial_index)

        self.deg_plot.clear()
        self._make_deg_ethogram_plot(trial_index=trial_index)

    def show(self):
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                  self.trial_selector]),
            self.plot.show(),
            self.deg_plot.show()])

