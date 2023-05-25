import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import HBox, VBox, Select

from fastplotlib import ImageWidget
from warnings import warn
from mesmerize_core.arrays import LazyVideo
from fastplotlib import Plot
from scipy.io import loadmat
from typing import *
import numpy as np


class BehaviorVizContainer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            local_parent_path: str,
            start_index: int = 0
    ):
        self._dataframe = dataframe
        self.local_parent_path = local_parent_path

        hide_columns = ["mat_file",
                        "session_vids",
                        "notes"]

        columns = dataframe.columns

        default_widths = {
            'animal_id': 200,
            'session_id': 200
        }

        df_show = self._dataframe[[c for c in columns if c not in hide_columns]]

        self.datagrid = DataGrid(
            df_show,
            selection_mode="cell",
            layout={"height": "250px", "width": "750px"},
            base_row_size=24,
            index_name="index",
            column_widths=default_widths)

        self.current_row: int = start_index
        self.trial_selector = None
        self.image_widget = None
        self.plot = None

        # initialize the ethogram plot and imagewidget

        self.datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )
        self._set_trial_selector(self.current_row)

        self.datagrid.observe(self._row_changed, names="selections")

    def _get_selection_row(self) -> Union[int, None]:
        r1 = self.datagrid.selections[0]["r1"]
        r2 = self.datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        index = self.datagrid.get_visible_data().index[r1]

        return index

    def _row_changed(self, *args):
        index = self._get_selection_row()

        if index is None or self.current_row == index:
            return

        self.current_row = index
        self._set_trial_selector(index)

    def _set_trial_selector(self, index):
        """Creates trial selector widget for a given session."""
        row = self._dataframe.iloc[index]
        options = [item.stem for item in row['session_vids']]

        if self.trial_selector is None:
            self.trial_selector = Select(options=options)
            self.trial_selector.observe(self._trial_change, "value")
        else:
            self.trial_selector.options = options

        self.selected_trial_ix = int(self.trial_selector.value.split('_v')[-1])

        if self.image_widget is None:
            self._make_image_widget()

        if self.plot is None:
            self._make_ethogram_plot()

    def _make_image_widget(self):
        """
        Instantiates image widget.
        """
        row = self._dataframe.iloc[self.current_row]
        vid_path = row['session_vids'][int(self.selected_trial_ix)]

        if self.image_widget is None:
            self.image_widget = ImageWidget(data=LazyVideo(vid_path))

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row]
        session_dir = self.local_parent_path.joinpath(row['animal_id']).joinpath(row['session_id'])

        if self.plot is None:
            self.plot = Plot()

        ethogram_shape = self._get_ethogram_shape(session_dir)
        eth_dtype = self._get_ethogram(0, list(session_dir.glob("*.mat"))[0])[0].dtype
        eth_heatmap = self.plot.add_heatmap(data=np.zeros(ethogram_shape, dtype=eth_dtype))
        eth_selector = eth_heatmap.add_linear_selector()

        eth_selector.selection.add_event_handler(self.ethogram_event_handler)

    def ethogram_event_handler(self, ev):
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

        session_path = self.local_parent_path.joinpath(row['animal_id']).joinpath(row['session_id'])
        selected_video = session_path.joinpath(self.trial_selector.value).with_suffix('.avi')

        self.image_widget._data = [LazyVideo(selected_video)]
        self.image_widget.current_index["t"] = 0
        self.image_widget.sliders["t"].value = 0
        self.image_widget.plot.graphics[0].data = self.image_widget._data[0][0]

        hm_data = self._get_ethogram(int(selected_video.stem.split('_v')[-1]), row['mat_file'])[0]
        self.plot.graphics[0].data[:hm_data.shape[0], :hm_data.shape[1]] = hm_data

    def _get_ethogram_shape(self, session_dir) -> Tuple[int, int]:
        """
        Gets the shape of the largest ethogram in order to allocate data buffer.
        """
        d0, d1 = (0, 0)
        for o in self.trial_selector.options:
            ix = int(o[-3:]) - 1
            eth = self._get_ethogram(ix, list(session_dir.glob("*.mat"))[0])[0].shape
            d0, d1 = (max(eth[0], d0), max(eth[1], d1))
        return d0, d1

    def _get_ethogram(self, trial_index: int, mat_path):
        """
        Returns the ethogram for a given trial in a session.
        """
        m = loadmat(mat_path)
        behaviors = sorted([b.split('_')[0] for b in m['data'].dtype.names if 'scores' in b])

        all_behaviors = [
            "Lift",
            "Handopen",
            "Grab",
            "Sup",
            "Atmouth",
            "Chew"
        ]

        sorted_behaviors = [b for b in all_behaviors if b in behaviors]

        ethograms = []

        mat_trial_index = np.argwhere(m["data"]["trial"].ravel() == (trial_index + 1))
        # Trial not found in JAABA data
        if mat_trial_index.size == 0:
            return False

        mat_trial_index = mat_trial_index.item()

        for b in sorted_behaviors:
            behavior_index = m['data'].dtype.names.index(f'{b}_postprocessed')
            ethograms.append(m['data'][mat_trial_index][0][behavior_index])

        sorted_behaviors = [b.lower() for b in sorted_behaviors]

        return np.hstack(ethograms).T, sorted_behaviors

    def show(self):
        """
        Shows the widget
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                  self.trial_selector]),
            self.plot.show()])
