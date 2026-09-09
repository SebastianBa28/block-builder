'''plothebi.py

   Plot the /joint_states and /joint_commands recorded in the ROS2 bag.

   Usage:
     plothebi <bagname> <joints>
   where
     <bagname> is either 'latest' or the name of a bag
     <joints>  is either 'all' or joint numbers or joint names
'''

import rclpy
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from sensor_msgs.msg            import JointState


######################################################################
# Grab the shared utility functions
from rosbag2_py                 import SequentialReader
from rclpy.serialization        import deserialize_message

#
#  Grab the latest ROS Bag name
#
def latestBag():
    # Report.
    print("Looking for latest ROS bag...")
        
    # Look at all bags, making sure we have at least one!
    dbfiles = glob.glob('*/metadata.yaml')
    if not dbfiles:
        raise FileNotFoundError('Unable to find a ROS2 bag')

    # Grab the modification times and the index of the newest.
    dbtimes = [os.path.getmtime(dbfile) for dbfile in dbfiles]
    i = dbtimes.index(max(dbtimes))

    # Select and return the newest.
    return os.path.dirname(dbfiles[i])

#
#   ROS Bag Reader
#
#   This is a wrapper to make accessing the bags a little easier
class BagReader:
    def __init__(self, bagname, verbose=True):
        # Set up the BAG reader.
        self.reader = SequentialReader()
        try:
            self.reader.open_uri(bagname)
        except Exception as e:
            print("Unable to read the ROS bag '%s'!" % bagname)
            print("Does it exist and WAS THE RECORDING Ctrl-c KILLED?")
            raise OSError("Error reading bag - did recording end?") from None

        # Report the contained topics and types:
        if verbose:
            print("The bag contains messages for:")
            for x in self.reader.get_all_topics_and_types():
                print("  topic %-20s of type %s" % (x.name, x.type))

    def close(self):
        # Close the reader's file.
        self.reader.close()

    def t0(self):
        # Get the starting time.
        return self.reader.get_metadata().starting_time.nanoseconds * 1e-9

    def msgs(self, topicname, topictype):
        # Restart the reader at the beginning of the file.
        self.reader.seek(0)

        # Pull out the relevant messages.
        msglist = []
        while self.reader.has_next():
            # Grab a message.
            (topic, rawdata, timestamp) = self.reader.read_next()

            # Pull out the deserialized message.
            if topic == topicname:
                msglist.append(deserialize_message(rawdata, topictype))

        # Return the list.
        return msglist

######################################################################

#
#  Extract Joint Data
#
def jointdata(msgs, t0, jointnames):
    # Make sure we have data
    if not msgs:
        raise ValueError("No data!")

    # Grab the names (assuming all will be the same).
    names = msgs[0].name

    # Grab the time.
    sec  = np.array([msg.header.stamp.sec     for msg in msgs])
    nano = np.array([msg.header.stamp.nanosec for msg in msgs])
    t    = sec + nano*1e-9 - t0

    # Grad the data.
    try:
        pos = np.array([msg.position for msg in msgs])
        vel = np.array([msg.velocity for msg in msgs])
        eff = np.array([msg.effort   for msg in msgs])
    except:
        raise ValueError("Data has inconsistent sizing")

    # Make sure we have data.
    (M,N) = (len(msgs), len(names))
    if np.shape(pos)[1] == 0:  pos = np.full((M,N), np.nan)
    if np.shape(vel)[1] == 0:  vel = np.full((M,N), np.nan)
    if np.shape(eff)[1] == 0:  eff = np.full((M,N), np.nan)
    if np.shape(pos)[1] != N: raise ValueError("Pos data not %d joints" % N)
    if np.shape(vel)[1] != N: raise ValueError("Vel data not %d joints" % N)
    if np.shape(eff)[1] != N: raise ValueError("Eff data not %d joints" % N)

    # Extract the specified joints.
    if jointnames[0] != 'all':
        # Loop over all jointnames
        indices = []
        for jointname in jointnames:
            # Grab the joint index/name.
            try:
                index = int(jointname)
                try:
                    jointname = names[index]
                except Exception:
                    raise ValueError("Joint %d out of range 0...%d" %
                                     (index, N))
            except Exception:
                try:
                    index = names.index(jointname)
                except Exception:
                    raise ValueError("Joint '%s' not in known joints %s" %
                                     (jointname, str(names)))

            # Append of the indices:
            indices.append(index)

        # Limit the data.
        print(f"Limiting data to indices {indices}")
        names = [names[index] for index in indices]
        i     = np.array(indices)
        pos   = pos[:,i[i<N]]
        vel   = vel[:,i[i<N]]
        eff   = eff[:,i[i<N]]

    # Return the data.    
    return (names, t, pos, vel, eff)


#
#  Plot the Joint Actual and Command Data
#
def plotboth(actmsgs, cmdmsgs, t0, title, jointnames=['all'], figsize=(15, 12)):
    # Process the actual and command messages.
    (nact, tact, pact, vact, eact) = jointdata(actmsgs, t0, jointnames)
    (ncmd, tcmd, pcmd, vcmd, ecmd) = jointdata(cmdmsgs, t0, jointnames)

    # Make sure the names match!
    if nact != ncmd:
        raise ValueError("Joint names in actual/command data do not match!")

    # Re-zero the start time.
    tstart = min(min(tact), min(tcmd))
    print("Starting at time ", tstart)
    tact = tact - tstart
    tcmd = tcmd - tstart

    # Create a figure to plot pos/vel/eff vs. t
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    # Plot the data in the subplots.
    axs[0].plot(tact, pact, linestyle='-' )
    axs[0].set_prop_cycle(None)
    axs[0].plot(tcmd, pcmd, linestyle='--')
    axs[0].set(ylabel='Position (rad)')

    axs[1].plot(tact, vact, linestyle='-' )
    axs[1].set_prop_cycle(None)
    axs[1].plot(tcmd, vcmd, linestyle='--')
    axs[1].set(ylabel='Velocity (rad/sec)')

    axs[2].plot(tact, eact, linestyle='-' )
    axs[2].set_prop_cycle(None)
    axs[2].plot(tcmd, ecmd, linestyle='--')
    axs[2].set(ylabel='Effort (Nm)')

    # Connect the time.
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])
    axs[2].set(xlabel='Time (sec)')

    # Add the title and legend.
    fig.suptitle(title)
    axs[0].legend(nact, ncol=len(nact),
                  loc='lower center', bbox_to_anchor=(0.5, 1.0))

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


#
#  Plot the Joint Data (Actual OR Command)
#
def plotsolo(jntmsgs, t0, title, jointnames=['all'], figsize=(15, 12)):
    # Process the joint messages.
    (n, t, p, v, e) = jointdata(jntmsgs, t0, jointnames)

    # Re-zero time.
    tstart = min(t)
    print("Starting at time ", tstart)
    t = t - tstart

    # Create a figure to plot pos/vel/eff vs. t
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    # Plot the data in the subplots.
    axs[0].plot(t, p)
    axs[0].set(ylabel='Position (rad)')
    axs[1].plot(t, v)
    axs[1].set(ylabel='Velocity (rad/sec)')
    axs[2].plot(t, e)
    axs[2].set(ylabel='Effort (Nm)')

    # Connect the time.
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])
    axs[2].set(xlabel='Time (sec)')
    
    # Add the title and legend.
    fig.suptitle(title)
    axs[0].legend(n, ncol=len(n),
                  loc='lower center', bbox_to_anchor=(0.5, 1.0))

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


#
#  Main Code
#
def main():
    # Grab the arguments.
    bagname    = 'latest' if len(sys.argv) < 2 else sys.argv[1]
    jointnames = ['all']  if len(sys.argv) < 3 else sys.argv[2:]

    # Check for the latest ROS bag:
    if bagname == 'latest':
        bagname = latestBag()

    # Report.
    print("Reading ROS bag:   " + str(bagname))
    print("Processing joints: " + str(jointnames))

    # Set up the BAG reader, grab the time and messages.
    reader = BagReader(bagname)
    t0     = reader.t0() - 0.01

    actmsgs = reader.msgs('/joint_states',   JointState)
    cmdmsgs = reader.msgs('/joint_commands', JointState)

    # Process the actual/command joint data
    if actmsgs and cmdmsgs:
        print("Plotting actual/command data...")
        title = "Actual/Command Data in '%s'" % bagname
        plotboth(actmsgs, cmdmsgs, t0, title, jointnames)

    elif actmsgs:
        print("Plotting actual data...")
        title = "Actual Data in '%s'" % bagname
        plotsolo(actmsgs, t0, title, jointnames)
    elif cmdmsgs:
        print("Plotting command data...")
        title = "Command Data in '%s'" % bagname
        plotsolo(cmdmsgs, t0, title, jointnames)

    else:
        raise ValueError("No joint data!")

    import os
    from datetime import datetime
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(file_dir, '../../../data/plots/', 'plot.png')
    print(f'Save Path: {os.path.normpath(save_path)}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()
