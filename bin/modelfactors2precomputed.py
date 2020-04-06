#!/usr/bin/env python3

import pandas

import sys


def usage():
    print("Usage: {} <input.csv>".format(sys.argv[0]))


def main():
    if len(sys.argv) > 2:
        usage()
        return -1

    inputfile = "modelfactors.csv"
    if len(sys.argv) == 2:
        inputfile = sys.argv[1]

    try:
        inputdata = pandas.read_csv(
            inputfile, sep=";", index_col=0, comment="#"
        ).transpose()
    except FileNotFoundError:
        print("FATAL: {} not found".format(inputfile))
        return -1

    inputdata.columns.name = "Metrics"

    mapper = {m: m.title() for m in inputdata.columns}
    replacements = {
        "Ipc": "IPC",
        "Scalability": "Scaling",
        "Load Balance": "MPI Load Balance",
        "Communication Efficiency": "MPI Communication Efficiency",
        "Transfer Efficiency": "MPI Transfer Efficiency",
        "Serialization Efficiency": "MPI Serialisation Efficiency",
    }
    for rin, rout in replacements.items():
        mapper = {k: v.replace(rin, rout) for k, v in mapper.items()}

    inputdata.rename(columns=mapper, inplace=True)

    for column in inputdata.columns:
        if column in ["Speedup", "Average IPC", "Average Frequency (Ghz)"]:
            continue
        inputdata[column] = inputdata[column] / 100

    inputdata["Number of Processes"] = inputdata.index
    inputdata["Threads per Process"] = 1
    inputdata["Total Threads"] = inputdata.index
    inputdata["Hybrid Layout"] = ["{}x1".format(i) for i in inputdata.index]

    inputdata.to_csv("precomputed.csv")


if __name__ == "__main__":
    main()
