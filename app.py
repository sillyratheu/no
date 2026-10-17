import streamlit as st
import subprocess
import os
import time

st.set_page_config(page_title="NotRa VNC Desktop", layout="wide")

st.title("🖥️ NotRa Lightweight VNC Desktop (XFCE)")

VNC_PASS = "12345678"
VNC_PORT = 5901

def setup_and_start_vnc():
    vnc_dir = os.path.expanduser("~/.vnc")
    os.makedirs(vnc_dir, exist_ok=True)
    
    # Set VNC password
    passwd_file = os.path.join(vnc_dir, "passwd")
    if not os.path.exists(passwd_file):
        subprocess.run(f"echo '{VNC_PASS}' | vncpasswd -f > {passwd_file}", shell=True)
        os.chmod(passwd_file, 0o600)

    # Configure xstartup for lightweight XFCE4
    xstartup_file = os.path.join(vnc_dir, "xstartup")
    if not os.path.exists(xstartup_file):
        with open(xstartup_file, "w") as f:
            f.write("#!/bin/sh\n")
            f.write("xrdb $HOME/.Xresources\n")
            f.write("startxfce4 &\n")
        os.chmod(xstartup_file, 0o755)

    # Start VNC Server on Display :1 (Port 5901)
    check_vnc = subprocess.run("pgrep Xtightvnc", shell=True)
    if check_vnc.returncode != 0:
        subprocess.run("vncserver :1 -geometry 1280x720 -depth 16", shell=True)

# Run VNC server initialization
setup_and_start_vnc()

st.success("✅ VNC Server is Running!")

st.subheader("🔑 VNC Connection Details")
st.code(f"Internal VNC Port: {VNC_PORT}\nPassword: {VNC_PASS}", language="text")

st.info("💡 Lightweight Midori browser is installed inside XFCE desktop so RAM usage stays low.")

# Keep alive indicator for NotRa Uptime
st.markdown("---")
st.write("🟢 Server status: **Active 24/7 via NotRa Uptime Dashboard**")
