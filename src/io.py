import pathlib
import warnings
import os
import datetime
import numpy as np
import json
import string
import pandas as pd
import random

# TODO: write uuid option


def current_time():
    """
    Returns current date and time in a consistent format, used for monitoring long-running measurements

    :return: current date and time
    :rtype: str
    """
    return datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")


class IO:
    """
    The IO class encapsulates all saving/loading features of data, figures, etc.
    This provides consistent filetypes, naming conventions, etc.

    Typical usage:
        io = IO(path=r"\path\to\data")
        io.load_txt(filename="filename.txt")

    or
        io = IO.create_new_save_folder(folder="subfolder", include_date=True, include_uuid=True)
        io.save_df(df, filename="dataframe.txt")
    """

    # default save path always points to `data/` no matter where this repository is located
    default_path = pathlib.Path(__file__).parent.parent.joinpath('data')

    def __init__(self, path=None, verbose=True):
        """
        Create an IO object

        :param path: path to the data folder
        :type path: str
        :param verbose: if True, IO actions are is printed to the terminal. If False, the prints are omitted
        :type verbose: bool
        :return: function returns nothing
        :rtype: None
        """
        self.verbose = verbose

        # set default path to a 'data' folder
        if path is None:
            path = self.default_path
        if type(path) is str:
            path = pathlib.Path(path)

        self.path = pathlib.Path(path)

    @classmethod
    def new_directory(cls, path=None, folder="",
                      include_date=False, include_id=False, verbose=True):
        """

        :param path: The parent folder.
        :type path: str (of pathlib.Path object)
        :param folder: The main, descriptive folder name.
        :type folder: str
        :param include_date: If True, add the date to the front of the path. Otherwise, do not add the date
        :type include_date: bool
        :param include_id: If True, add a random string of characters to the end of the path. Otherwise, do not
        :type include_id: bool
        :param verbose: If True, will print out the path of each saved/loaded file.
        :type verbose: bool
        :return: A new IO class instance
        :rtype: IO
        """
        if path is None:
            path = cls.default_path

        if type(path) is str:
            path = pathlib.Path(path)

        date = datetime.date.today().isoformat()
        if not folder:  # if empty string
            warnings.warn("No folder entered. Saving to a folder with a unique identifier")
            include_data, include_id, verbose = True, True, True

        if include_date:
            folder = date + " " + folder
        if include_id:
            folder = folder + " - " + "".join(random.choice(string.hexdigits) for _ in range(4))

        path = path.joinpath(folder)
        return cls(path=path, verbose=verbose)

    def save_json(self, variable, filename):
        """
        Save serialized python object into a json format, at filename

        :param variable: the object to save
        :type variable: serialized object
        :param filename: name of the file to which variable should be saved
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        self._save_json(variable, full_path)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_json(self, filename):
        """
        Load serialized python object from json

        :param filename: name of the file from which we are loading the object
        :type filename: str
        :return: the loaded object data
        :rtype: may vary
        """
        full_path = self.path.joinpath(filename)
        file = self._load_json(full_path)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_dataframe(self, df, filename):
        """
        Save a panda dataframe object to csv

        :param df: data contained in a dataframe
        :type df: panda dataframe
        :param filename: file to which data should be saved
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        df.to_csv(str(full_path), sep=',', index=False, header=True)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_dataframe(self, filename):
        """
        Load panda dataframe object from CSV

        :param filename: name of the file from which data should be loaded
        :type filename: str
        :return: dataframe data
        :rtype: panda dataframe
        """
        full_path = self.path.joinpath(filename)
        df = pd.read_csv(str(full_path), sep=",", header=0)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return df

    def load_timetags(self, filename):
        # TODO: docstring (am not 100% sure what the use case of this function is)
        full_path = self.path.joinpath(filename)
        data = np.loadtxt(str(full_path),  delimiter="\t")
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return data

    def save_figure(self, fig, filename):
        """
        Save a figure (image datatype can be specified as part of filename)

        :param fig: the figure containing the figure to save
        :type fig: matplotlib.figure
        :param filename: the filename to which we save a figure
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        if self.verbose:
            print(f"{current_time()} | Saved figure to {full_path} successfully.")

    def save_np_array(self, np_arr, filename):
        """
        Save numpy array to a text document

        :param np_arr: the array which we are saving
        :type np_arr: numpy.array
        :param filename: name of the text file to which we want to save the numpy array
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        np.savetxt(str(full_path), np_arr)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_np_array(self, filename, complex_vals=False):
        """
        Loads numpy array from a text document

        :param filename: name of the text file from which we want to load the numpy array
        :type filename: str
        :param complex_vals: True if we expect the numpy array to be complex, False otherwise
        :type complex_vals: bool
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        file = np.loadtxt(str(full_path), dtype=np.complex if complex_vals else np.float)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    @staticmethod
    def _save_json(variable, path):
        """
        Helper method for saving to json files
        """
        with open(path, 'w+') as json_file:
            json.dump(variable, json_file, indent=4)

    @staticmethod
    def _load_json(path):
        """
        Helper method for loading from json files
        """
        with open(path) as json_file:
            data = json.load(json_file)
        return data
