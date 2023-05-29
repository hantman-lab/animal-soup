import os.path

import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import HBox, VBox, Select

from fastplotlib import ImageWidget
from warnings import warn
from mesmerize_core.arrays import LazyVideo
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector
from scipy.io import loadmat
from typing import *
import numpy as np

from .batch_utils import get_parent_raw_data_path, PARENT_DATA_PATH, validate_path


ETHOGRAM_COLORS = {
    "lift": "b",
    "handopen": "green",
    "grab": "r",
    "sup": "cyan",
    "atmouth": "magenta",
    "chew": "yellow"
}


class BehaviorVizContainer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
            deg_comparison: bool = False
    ):
        self._dataframe = dataframe
        self.local_parent_path = get_parent_raw_data_path()
        self.deg_comparison = deg_comparison

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
        self.deg_plot = None

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

        if self.deg_plot is None and self.deg_comparison:
            self._make_deg_ethogram_plot()

    def _make_image_widget(self):
        """
        Instantiates image widget.
        """
        row = self._dataframe.iloc[self.current_row]
        vid_path = row['session_vids'][int(self.selected_trial_ix)]

        if self.image_widget is None:
            self.image_widget = ImageWidget(data=LazyVideo(vid_path))

    def _make_ethogram_plot(self, trial_index: int = 0):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row]
        session_dir = self.local_parent_path.joinpath(row['animal_id']).joinpath(row['session_id'])

        if self.plot is None:
            self.plot = Plot(size=(500, 100))

        # ethogram_shape = self._get_ethogram_shape(session_dir)
        self.ethogram_array, self.behaviors = self._get_ethogram(trial_index, list(session_dir.glob("*.mat"))[0])

        #colors = list(map(ETHOGRAM_COLORS.get, self.behaviors))

        y_bottom = 0
        for i, b in enumerate(self.behaviors):
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

        self.ethogram_selector = LinearSelector(
            selection=0,
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            parent=lg,
            end_points=(y_bottom, y_pos),
        )

        self.plot.add_graphic(self.ethogram_selector)

        self.ethogram_selector.selection.add_event_handler(self.ethogram_event_handler)
        self.plot.camera.maintain_aspect = False
        self.plot.auto_scale()

    def _make_deg_ethogram_plot(self, trial_index: int = 0):
        row = self._dataframe.iloc[self.current_row]
        session_dir = self.local_parent_path.joinpath(row['animal_id']).joinpath(row['session_id'])

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

        self.deg_ethogram_selector.selection.add_event_handler(self.deg_ethogram_event_handler)
        self.deg_plot.camera.maintain_aspect = False
        self.deg_plot.auto_scale()

    def deg_ethogram_event_handler(self, ev):
        """
        Event handler called for linear selector.
        """
        ix = ev.pick_info["selected_index"]
        self.image_widget.sliders["t"].value = ix

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

        trial_index = int(selected_video.stem.split('_v')[-1]) - 1
        self.plot.clear()
        self._make_ethogram_plot(trial_index=trial_index)
        if self.deg_comparison:
            self.deg_plot.clear()
            self._make_deg_ethogram_plot(trial_index=trial_index)

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
        if self.deg_comparison:
            return VBox([
                self.datagrid,
                HBox([self.image_widget.show(),
                      self.trial_selector]),
                self.plot.show(),
                self.deg_plot.show()])
        else:
            return VBox([
                self.datagrid,
                HBox([self.image_widget.show(),
                      self.trial_selector]),
                self.plot.show()])


@pd.api.extensions.register_dataframe_accessor("behavior")
class BehaviorDataFrameVizExtension:
    def __init__(self, df):
        self._dataframe = df

    def view(
            self,
            start_index: int = 0,
            deg_comparison: bool = False
    ):
        container = BehaviorVizContainer(
            dataframe=self._dataframe,
            start_index=start_index,
            deg_comparison=deg_comparison
        )

        return container

    def add_item(
            self,
            animal_id: str,
            session_id: Union[str, None] = None):

        if get_parent_raw_data_path() is None:
            raise ValueError(
                "parent raw data path is not set, you must set it using:\n"
                "`set_parent_raw_data_path()`"
            )

        PARENT_DATA_PATH = get_parent_raw_data_path()

        animal_dir = PARENT_DATA_PATH.joinpath(animal_id)

        validate_path(animal_dir)

        if not os.path.exists(animal_dir):
            raise ValueError(
                f"animal_id path is not valid at: {animal_dir}"
            )

        if session_id is None:
            session_dirs = sorted(animal_dir.glob('*'))
            if len(session_dirs) == 0:
                raise ValueError("no sessions found in this animal dir")
            for session_dir in session_dirs:
                session_id = session_dir.stem
                mat_file = session_dir.joinpath('jaaba.mat')
                session_vids = sorted(session_dir.glob('*.avi'))

                s = pd.Series(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "mat_file": mat_file,
                        "session_vids": session_vids,
                        "notes": None
                    }
                )

                self._dataframe.loc[self._dataframe.index.size] = s

                self._dataframe.to_hdf(self._dataframe.paths.get_df_path(), key='df')
        else:
            session_dir = animal_dir.joinpath(session_id)
            mat_file = session_dir.joinpath('jaaba.mat')
            session_vids = sorted(session_dir.glob('*.avi'))
            s = pd.Series(
                {
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "mat_file": mat_file,
                    "session_vids": session_vids,
                    "notes": None
                }
            )

            self._dataframe.loc[self._dataframe.index.size] = s

            self._dataframe.to_hdf(self._dataframe.paths.get_df_path(), key='df')


