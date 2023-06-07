import pandas as pd
from ipywidgets import HBox, VBox, Select, Button, Layout, RadioButtons

from mesmerize_core.arrays import LazyVideo
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector
from scipy.io import loadmat
import numpy as np
from ._behavior import BehaviorVizContainer

ETHOGRAM_COLORS = {
    "lift": "b",
    "handopen": "green",
    "grab": "r",
    "sup": "cyan",
    "atmouth": "magenta",
    "chew": "yellow"
}

class EthogramVizContainer(BehaviorVizContainer):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 start_index: int = 0,
                 ):
        super(EthogramVizContainer, self).__init__(
            dataframe=dataframe,
            start_index=start_index
        )

        self.plot = None

        self.trial_button = Button(value=False, disabled=False, icon='hand-pointer',
                                       layout=Layout(width='auto'), tooltip='select trial as train')

        self.ethogram_button = RadioButtons(options=['hand-labels', 'jabba-pred'],
                                         value='hand-labels',
                                                description='Ethogram Options:',
                                                disabled=False
)

        self.trial_button.on_click(self.add_selected_trial)
        self.ethogram_button.observe(self.change_ethogram, 'value')

        self._make_ethogram_plot()

    def _make_ethogram_plot(self, trial_index: int = 0):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row]
        session_dir = self.local_parent_path.joinpath(row['animal_id'], (row['session_id']))

        if self.plot is None:
            self.plot = Plot(size=(500, 100))

        self.ethogram_array, self.behaviors = self._get_ethogram(trial_index, list(session_dir.glob("*.mat"))[0], ethogram_type=self.ethogram_button.value)

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

        session_path = self.local_parent_path.joinpath(row['animal_id'], row['session_id'])
        selected_video = session_path.joinpath(self.trial_selector.value).with_suffix('.avi')

        self.image_widget._data = [LazyVideo(selected_video)]
        self.image_widget.current_index["t"] = 0
        self.image_widget.sliders["t"].value = 0
        self.image_widget.plot.graphics[0].data = self.image_widget._data[0][0]

        trial_index = int(selected_video.stem.split('_v')[-1]) - 1
        self.plot.clear()
        self._make_ethogram_plot(trial_index=trial_index)

    def _get_ethogram(self, trial_index: int, mat_path, ethogram_type: str):
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

        if ethogram_type == 'hand-labels':
            for b in sorted_behaviors:
                behavior_index = m['data'].dtype.names.index(f'{b}_labl_label')
                row = m['data'][mat_trial_index][0][behavior_index]
                row[row == -1] = 0
                ethograms.append(row)
        else:
            for b in sorted_behaviors:
                behavior_index = m['data'].dtype.names.index(f'{b}_postprocessed')
                row = m['data'][mat_trial_index][0][behavior_index]
                row[row == -1] = 0
                ethograms.append(row)

        sorted_behaviors = [b.lower() for b in sorted_behaviors]

        return np.hstack(ethograms).T, sorted_behaviors

    def add_selected_trial(self, obj):
        row = self._dataframe.iloc[self.current_row]

        selected_trial = self.trial_selector.value

        if self.ethogram_button.value == 'hand-labels':
            if selected_trial not in row["hand_training_trials"]:
                row["hand_training_trials"].append(selected_trial)
        else:
            if selected_trial not in row["jabba_training_trials"]:
                row["jabba_training_trials"].append(selected_trial)

        self._dataframe.to_hdf(self._dataframe.paths.get_df_path(), key='df')

    def change_ethogram(self, obj):
        # change in ethogram option should regenerate current ethogram plot
        row = self._dataframe.iloc[self.current_row]

        session_path = self.local_parent_path.joinpath(row['animal_id'], row['session_id'])
        selected_video = session_path.joinpath(self.trial_selector.value).with_suffix('.avi')

        trial_index = int(selected_video.stem.split('_v')[-1]) - 1
        self.plot.clear()
        self._make_ethogram_plot(trial_index=trial_index)

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),

                  VBox([self.trial_selector,
                        self.trial_button,
                       self.ethogram_button])
        ]),
            self.plot.show()])