#!/usr/bin/enb python3

"""A minimal interface for managing IPython servers
"""

import os
import subprocess
from time import sleep

from notebook.notebookapp import list_running_servers


def get_cache_pid():

    try:
        with open(os.path.expanduser("~/.cache/pypopserver.pid")) as fh:
            pid = int(fh.readline().strip())
            return pid
    except (FileNotFoundError, ValueError):
        return None


def get_notebook_server_instance(try_use_existing=False):
    """Create a notebook server instance to use. Optionally attempting to re-use existing
    instances.
    """

    pid = get_cache_pid()

    servers = list_running_servers()

    # If we already have a server, use that
    for server in servers:
        if server["pid"] == pid:
            return (server, None)

    # Otherwise, if we are allowed, try to piggyback on another session
    if try_use_existing and servers:
        return (servers[0], None)

    # Fine, I'll make my own server, with blackjack, and userhooks!
    try:
        server_process = subprocess.Popen(["jupyter", "notebook", "--no-browser"])
    except OSError as err:
        raise RuntimeError("Failed to start server: {}".format(err))
    print("Started Jupyter Notebook server pid {}".format(server_process.pid))
    # wait for 1 second for server to come up
    sleep(1)

    server = None
    for retry in range(5):
        try:
            server = {s["pid"]: s for s in list_running_servers()}[server_process.pid]
            break
        except KeyError:
            # Sleep for increasing times to give server a chance to come up
            sleep(5)

    if server:
        return (server, server_process)

    # Don't leave orphans!
    server_process.kill()
    raise RuntimeError("Failed to acquire server instance after 25s")


def construct_nb_url(server_info, nb_path):

    nb_dir = server_info["notebook_dir"]

    if os.path.isabs(nb_dir):
        nb_relpath = os.path.relpath(nb_path, nb_dir)
        if nb_relpath.startswith(".."):
            raise ValueError(
                "Requested notebook file path is not in the notebook server "
                "directory ({}).".format(nb_path, nb_dir)
            )
    else:
        nb_relpath = nb_path.lstrip(".").lstrip("/")

    if server_info["token"]:
        return "{}notebooks/{}?token={}".format(
            server_info["url"], nb_relpath, server_info["token"]
        )
    else:
        return "{}notebooks/{}".format(server_info["url"], nb_relpath)
