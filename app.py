import streamlit as st
import subprocess
import os

st.set_page_config(page_title="NotRa VNC Desktop", layout="wide")

st.title("🖥️ NotRa Lightweight VNC Desktop (XFCE)")

VNC_PASS = "12345678"
VNC_PORT = 5901

def setup_and_start_vnc():
    vnc_dir = os.path.expanduser("~/.vnc")
    os.makedirs(vnc_dir, exist_ok=True)
    
    passwd_file = os.path.join(vnc_dir, "passwd")
    
    # Properly set VNC password using standard printf
    pass_cmd = f"printf '{VNC_PASS}\n{VNC_PASS}\n' | vncpasswd"
    subprocess.run(pass_cmd, shell=True)
    
    if os.path.exists(passwd_file):
        os.chmod(passwd_file, 0o600)

    # Configure xstartup for XFCE4
    xstartup_file = os.path.join(vnc_dir, "xstartup")
    with open(xstartup_file, "w") as f:
        f.write("#!/bin/sh\n")
        f.write("xrdb $HOME/.Xresources\n")
        f.write("startxfce4 &\n")
    os.chmod(xstartup_file, 0o755)

    # Kill any existing stuck VNC server sessions
    subprocess.run("vncserver -kill :1", shell=True, capture_output=True)

    # Start VNC Server on Display :1 (Port 5901)
    subprocess.run(f"vncserver :1 -geometry 1280x720 -depth 16", shell=True)

# Run VNC server initialization
setup_and_start_vnc()

st.success("✅ VNC Server is Running!")

st.subheader("🔑 VNC Connection Details")
st.code(f"Password: {VNC_PASS}", language="text")

st.markdown("---")
st.write("🟢 Server status: **Active 24/7 via NotRa Uptime Dashboard**")
