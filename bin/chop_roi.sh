#!/bin/bash

# This script assumes the paramedir executable (found in the wxparaver/bin directory)
# is in your PATH environment variable.

# Pass in all prv files as arguments.
# If no files are provided, exit
if [ $# -eq 0 ]
  then
  echo "No arguments supplied"
  exit 1
fi

# Names for XML files the scrip generates. These don't need to be changed as the files
# are created by this script (i.e. they are not prerequisite separate files).
filter_file=filter.xml
cutter_file=cutter.xml

# Temporary file to hold just the events from a prv file. 
tmp_events=events.txt

# Directory that will be created to hold the chopped traces
ROI_dir=ROI_chops

# For every input prv file...
for f in "$@"
do
  ( cd $(dirname $f)
    f=$(basename $f)
    # Create the XML filter file that paramedir will use to extract just the
    # 40000012 events from the trace files. These are the ones that occur
    # when tracing is enabled/disabled via the API. The filter file is the same
    # for all input prv files.
    printf "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" > $filter_file
    printf "<config>\n  <filter>\n    <discard_states>1</discard_states>\n" >> $filter_file
    printf "    <discard_events>0</discard_events>\n" >> $filter_file
    printf "    <discard_communications>1</discard_communications>\n" >> $filter_file
    printf "    <states>Running\n" >> $filter_file
    printf "      <min_state_time>0</min_state_time>\n" >> $filter_file
    printf "    </states>\n" >> $filter_file
    printf "    <types>\n      <type>40000012</type>\n" >> $filter_file
    printf "    </types>\n" >> $filter_file
    printf "    <comms>0</comms>\n" >> $filter_file
    printf "  </filter>\n</config>\n" >> $filter_file

    # Check if the prv file is gzipped, if it is unzip so we can access the raw numbers and 
    # change the filevariable to use the new name by removing .gz
    if [[ $f == *".gz" ]]; then
	echo "found gzip file"
	gunzip $f
	tmp=${f%.gz}
	f=$tmp
	echo "unzipped filename is: " $f
    fi

    # Extract the number of ranks/threads in the prv file. This is found in the 4th column
    # of the 2nd line (the columns are separated by ":")
    ncores=`head -2 $f | tail -1 | awk -F: '{print $4}'`
    echo $f $ncores

    # Call paramedir with the filter file to generate a new trace prv that contains only the
    # events that correspond to enabling/disabling tracing via the API, and then strip the header
    # line from this prv and stash the actual events in a temp file. The new trace prv is then
    # deleted as we don't need it anymore.
    paramedir --filter $f $filter_file
    grep -v 'c' *.filter1.prv | grep -v '#' > $tmp_events
    rm *.filter1.*

    # The bit of the trace we're interested in starts after tracing is first disabled. Because
    # tmp_events only contains enable/disable tracing events, the first ncores event will be
    # each rank/thread emitting a "tracing disabled" event. Therefore the time we're going to
    # start the cut at is immediately after the last one of these events (the "+1" in the print
    # statement adds 1ns to the last time).
    #
    # The logic for extracting this assumes each event is on a separate line and that the lines
    # are ordered by increasing time, and that enable/disable tracing events are not
    # intermingled (i.e. the trace has the disable events, all the enable events and finally
    # all the disable events). 
    start_time=`head -${ncores} $tmp_events | tail -1 | awk -F: '{print $6+1}'`


    # The bit of the trace we're interested in ends after tracing is disabled again. As with the
    # start time we can find this by counting lines in the trace file. There will be ncores lines
    # of initial "tracing disabled" events, a further ncores lines of "tracing enabled" events
    # (when the RoI starts) and then a further ncores lines of "tracing disabled" events (at the
    # end of the RoI). We therefore take the end time for cutting as just after the last "tracing
    # disabled" event in the second group of such events.
    #
    # Again the logic for extracting this assumes each event is on a separate line, that the
    # lines are ordered by increasing time, and that enable/disable tracing events are not
    # intermingled (i.e. the trace has the disable events, all the enable events and finally
    # all the disable events).
    endline=$(( $ncores *3 ))
    end_time=`head -${endline} $tmp_events | tail -1 | awk -F: '{print $6+1}'`

    echo -e "ncores\tstart\tstop\tendline"
    echo -e "$ncores\t$start_time\t$end_time\t$endline"

    # Create the XML cutter file to be used by paramedir to actually extract the RoI from
    # the full trace. This uses the start and end times identified from the filtered file.
    # We need to remove the first state and the last state because of the times we've chosen:
    # both times are in the "tracing disabled" states that surrounds the RoI.
    printf "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" > $cutter_file
    printf "<config>\n  <cutter>\n" >> $cutter_file
    printf "    <max_trace_size>0</max_trace_size>\n" >> $cutter_file
    printf "    <by_time>1</by_time>\n " >> $cutter_file
    printf "    <minimum_time>%d</minimum_time>\n" $start_time >> $cutter_file
    printf "    <maximum_time>%d</maximum_time>\n" $end_time >> $cutter_file
    printf "    <minimum_time_percentage>0</minimum_time_percentage>\n" >> $cutter_file
    printf "    <maximum_time_percentage>100</maximum_time_percentage>\n" >> $cutter_file
    printf "    <original_time>0</original_time>\n" >> $cutter_file
    printf "    <break_states>0</break_states>\n" >> $cutter_file
    printf "    <remove_first_states>1</remove_first_states>\n" >> $cutter_file
    printf "    <remove_last_states>1</remove_last_states>\n" >> $cutter_file
    printf "    <keep_events>1</keep_events>\n" >> $cutter_file
    printf "  </cutter>\n</config>\n" >> $cutter_file

    # Call paramedir with the cutter file to produce the RoI trace for this number of cores.
    outfile=roi_$f
    paramedir -c $f $cutter_file -o $outfile

    # Delete the temporary file containing just the tracing enabled/disabled events.
    rm $tmp_events

    # Finally delete the filter file and cutter file.
    rm $cutter_file $filter_file 
  )
done
