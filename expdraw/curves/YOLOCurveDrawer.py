import os
import matplotlib.pyplot as plt
import math
import pandas
from dataclasses import dataclass
from typing import Literal, Union, Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

path_folder = "/home/zhangfeihong/code/zza/yolov5/runs/train/coco_up"

path_pic = os.path.join(path_folder, "pics")
if not os.path.exists(path=path_pic):
    os.mkdir(path_pic)

child_folders = [name for name in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, name))]

tags_train = ['train/box_loss', 'train/obj_loss', 'train/cls_loss']
tags_val = ['val/box_loss', 'val/obj_loss', 'val/cls_loss']
tags_metric = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']
tags_lr = ['x/lr0', 'x/lr1', 'x/lr2']

tags_dict = {
    "tags_TrainLoss": tags_train, 
    "tags_ValLoss": tags_val, 
    "tags_metric": tags_metric, 
    "tags_lr": tags_lr
    }

def extract_data_tag(log_dir, tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    print(event_acc.Tags()['scalars'])
    if tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Tag {tag} not found in TensorBoard logs.")

    scalar_data = event_acc.Scalars(tag)
    steps = [x.step for x in scalar_data]
    values = [x.value for x in scalar_data]

    return steps, values

@dataclass
class UniformedTrainData:
    info: Dict[str, Dict[str, list]]
    steps: list
    log_dir: str

    @property
    def name(self):
        return os.path.basename(self.log_dir)

    """
    return: 
    {
        "info":{
            "tags 1": {"tag1": ...},
            ...,
        }
        "steps": list
        "log_dir": log_dir
    }
    """

def extract_data_uniformed_tbd(log_dir, **kwargs)->UniformedTrainData:
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    info = {}
    steps = None
    for key, val in tags_dict.items():
        temp = {}
        for tag in val:
            scalar_data = event_acc.Scalars(tag)
            if steps is None:
                steps = [x.step for x in scalar_data]
            values = [x.value for x in scalar_data]
            temp[tag] = values
        info[key] = temp
    assert steps is not None, "Steps not defined."
    return UniformedTrainData(info, steps, log_dir)

def extract_data_uniformed_csv(log_dir, **kwargs)->UniformedTrainData:
    df = pandas.read_csv(os.path.join(log_dir, "results.csv"))
    df.rename(columns=lambda x: x.strip(), inplace=True)
    steps = df["epoch"].to_list()
    info = {}
    for key, tag_list in tags_dict.items():
        temp = {}
        for tag in tag_list:
            temp[tag] = df[tag].to_list()
        info[key] = temp
    assert steps is not None, "Steps not defined."
    return UniformedTrainData(info, steps, log_dir)


def get_dirs(path_folder):
    ret = []
    for folder in os.listdir(path_folder):
        if os.path.isdir(os.path.join(path_folder, folder)):
            ret.append(folder)
    return ret

@dataclass
class PlotFigure:
    fig: plt.Figure
    axs: Dict[str, plt.Axes]
    key: str

    truncate:int=None
    alpha=0.6
    axe_legend=None
    is_legend=True

    def set_plot_config(self, **kwargs):
        if "truncate" in kwargs:
            self.truncate = kwargs["truncate"]
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]

    @staticmethod
    def prepare_axes(fig: plt.Figure, tags:list):
        edge_len = math.ceil(math.sqrt(len(tags)))
        ret = {}
        for i, tag in enumerate(tags):
            ax:plt.Axes = fig.add_subplot(edge_len, edge_len, i+1)
            ret[tag] = ax
            ax_name = tag.split("/")[-1]
            ax.set_title(ax_name)
        return ret

    def plotinfo(self, data:UniformedTrainData):
        steps = data.steps
        name = data.name
        last_idx = -1 if self.truncate is None else min(self.truncate, len(steps))
        values_dict = data.info[self.key]
        for key, val in values_dict.items():
            ax:plt.Axes = self.axs[key]
            ax.plot(steps[:last_idx], val[:last_idx], label=name, alpha=self.alpha)
            pass
        return

    def tidy(self, legend_method:Literal["normal", "outside"]="outside"):
        for key, val in self.axs.items():
            val.grid()
        self.fig.tight_layout()
        if self.is_legend:
            self.legend_outside(val)
        

    def legend_newax(self, ax):
        l = math.sqrt(len(self.axs))
        if math.ceil(l) > l:
            l = math.ceil(l)
            self.axe_legend = self.fig.add_subplot(l,l,l**2)
        else:
            self.axe_legend = ax
        handles, labels = ax.get_legend_handles_labels()
        self.axe_legend.legend(handles, labels, loc='best')

    def legend_outside(self, ax):
        handles, labels = ax.get_legend_handles_labels()

        # lower center
        self.fig.legend(handles, labels, loc='lower center', ncol=3)
        self.fig.subplots_adjust(bottom=0.15)

        # right center
        # self.fig.legend(handles, labels, loc='center right')
        # self.fig.subplots_adjust(right=0.8)

    def legend_normal(self, ax):
        self.axe_legend = ax
        self.axe_legend.legend(loc="best")

    def save_fig(self):
        fig = self.fig
        fig.savefig(os.path.join(path_pic, f"{self.key}.png"))

    pass

def plot_all(dirs:list, extract_data_uniformed=extract_data_uniformed_csv, **kwargs):
    
    # prepare fig dict
    fig_dict:Dict[str, PlotFigure] = {}
    for key, val in tags_dict.items():
        title = key.split("_")[-1]
        fig = plt.figure()
        fig.suptitle(title)
        pfig = PlotFigure(fig, PlotFigure.prepare_axes(fig, val), key)
        pfig.set_plot_config(**kwargs)
        # if key == "tags_metric":
        #     pfig.is_legend=False
        fig_dict[key] = pfig

    for dir in dirs:
        data = extract_data_uniformed(dir)
        for key in data.info.keys():
            pfig = fig_dict[key]
            pfig.plotinfo(data)

    for key, val in fig_dict.items():
        val.tidy()
        val.save_fig()
    return


if __name__ == "__main__":
    
    dirs = [
        # "AdamW_lre3",
        "AdamWRad_lre3",
        # "Adam_lre2", 
        "Adam_lre3", 
        # "Adam_lre53", 
        "Rad_lre2"
    ]
    dirs = [os.path.join(path_folder, dir) for dir in dirs]
    plot_all(dirs, truncate=300)

