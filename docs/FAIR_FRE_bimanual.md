# FAIR FRE Bimanual Deployment Guide

In NUC (tmr-nux)

```bash
cd murp
# start all control serve passward = safenrobust, will open a tmux (make sure there is no error)
./scripts/start_tmr_static.sh
```

In Host (tmr-host)

```bash
cd murp
# start host, passward = safenrobust, if the script doesn't work, go to nuc and manually unlock the arm under the `192.168.1.21`(left) `192.168.1.22`(right), unlock and then enable FCI, choose execution
./scripts/start_host.sh

# run script to control
cd chang
conda activate murp_env
source /opt/ros/humble/setup.bash
source ~/ws_rmw_zenoh/install/setup.bash
python murp_replay.py
```