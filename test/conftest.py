import gurobipy
import pytest

import gurobi_mock
import os
import glob
import logging
import datetime
from time import sleep

import alib.util as util


@pytest.fixture
def mock_gurobi(monkeypatch):
    monkeypatch.setattr(gurobipy, "LinExpr", gurobi_mock.MockLinExpr)
    monkeypatch.setattr(gurobipy, "Model", gurobi_mock.MockModel)


@pytest.fixture
def import_gurobi_mock():
    return gurobi_mock

#TODO: need a better way for cleaning up log file
#TODO: the following implementation is not xdist-safe
_log_directory = None
@pytest.fixture(scope="session", autouse=True)
def check_and_create_log_diretory(request):
    print("\n\nChecking whether directory {} exists...".format(util.ExperimentPathHandler.LOG_DIR))
    if not os.path.exists(util.ExperimentPathHandler.LOG_DIR):
        print("\tdid not exist, will create...".format(util.ExperimentPathHandler.LOG_DIR))
        os.mkdir(util.ExperimentPathHandler.LOG_DIR)
        print("\tcreated.".format(util.ExperimentPathHandler.LOG_DIR))
        _log_directory = util.ExperimentPathHandler.LOG_DIR
        #only if it was created, we remove it...

        def remove_log_directory():
            if _log_directory is not None:
                import shutil
                print("\n\nGoing to remove directory {}..".format(_log_directory))
                for logfile in glob.glob(_log_directory + "/*.log"):
                    print("\tremoving file {}..".format(logfile))
                    os.remove(logfile)
                print("\tremoving directoy.")
                os.rmdir(_log_directory)
                print("\tOK.")

        request.addfinalizer(remove_log_directory)
    else:
        print("\tdirectory exists; will be reused!")


def pytest_addoption(parser):
    log_help_text = 'Similar to log_file, but %%w will be replaced with a worker identifier.'
    parser.addini('worker_log_file', help=log_help_text)
    log_group = parser.getgroup('logging')
    log_group.addoption('--worker-log-file', dest='worker_log_file', help=log_help_text)

def pytest_configure(config):
    log_file = config.getoption('worker_log_file')
    if log_file is None:
        log_file = config.getini('worker_log_file')
    log_level_opt = config.getoption('log_level')
    if log_level_opt is None:
        log_level_opt = config.getini('log_level')
    if log_level_opt:
        log_level = getattr(logging, log_level_opt.upper(), None)
        if not log_level:
            raise RuntimeError('Invalid log-level option: {}'.format(log_level_opt))
    else:
        log_level = logging.DEBUG
    if log_file:
        worker_string = os.environ.get('PYTEST_XDIST_WORKER', '-1')
        numeric_worker_string = int("".join([a for a in worker_string if a.isdigit()]))
        if numeric_worker_string > -1:
            sleep(1)

        log_file = log_file.replace('%w', str(numeric_worker_string))

        counter = 0
        directory = "log_{}".format(datetime.datetime.now().strftime("%Y_%m_%d_%H"))
        actual_directory = directory + "_{:04d}".format(counter)

        while os.path.exists(actual_directory) and os.path.exists(os.path.join(actual_directory, log_file)):
            counter += 1
            actual_directory = directory + "_{:04d}".format(counter)
        if not os.path.exists(actual_directory):
            os.mkdir(actual_directory)

        logging.basicConfig(
            format='%(asctime)s %(name)s %(levelname)s %(message)s',
            filename=os.path.join(actual_directory,log_file),
            level=log_level)