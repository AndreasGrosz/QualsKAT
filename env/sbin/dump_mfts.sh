#!/bin/sh
#
# This simple script will collect outputs of ibroute for all switches
# on the subnet and drop it on stdout. It can be used for MFTs dump
# generation.
#

/home/res/projekte/kd0241-py/Q-KAT/AAUNaL/Beta-v1.2/env/sbin/dump_fts -M $@
echo ""
echo "*** WARNING ***: this command has been replaced by dump_fts -M"
echo ""
echo ""
