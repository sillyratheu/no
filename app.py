import streamlit as st
import subprocess
import os
import re
import time

st.set_page_config(page_title="NotRa VNC Desktop", layout="wide")

st.title("🖥️ NotRa VNC Desktop (XFCE + Bore Tunnel)")

VNC_PASS = "12345678"
LOCAL_VNC_PORT = 5901

@st.cache_resource
def setup_vnc_and_bore():
    # 1. Install system dependencies including dbus-x11
    install_cmd = """
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y \
        dbus-x11 \
        xfce4 \
        xfce4-goodies \
        tightvncserver \
        procps \
        net-tools \
        curl
    """
    subprocess.run(install_cmd, shell=True)

    # 2. Setup VNC Directory and Password
    vnc_dir = os.path.expanduser("~/.vnc")
    os.makedirs(vnc_dir, exist_ok=True)
    
    passwd_file = os.path.join(vnc_dir, "passwd")
    subprocess.run(f"printf '{VNC_PASS}\n{VNC_PASS}\n' | vncpasswd", shell=True)
    if os.path.exists(passwd_file):
        os.chmod(passwd_file, 0o600)

    # 3. Configure XFCE Startup to explicitly launch window manager and panel
    xstartup_file = os.path.join(vnc_dir, "xstartup")
    with open(xstartup_file, "w") as f:
        f.write("#!/bin/sh\n")
        f.write("unset DBUS_SESSION_BUS_ADDRESS\n")
        f.write("xrdb $HOME/.Xresources\n")
        f.write("xfsettingsd &\n")
        f.write("xfwm4 --replace &\n")
        f.write("xfce4-panel &\n")
        f.write("exec startxfce4\n")
    os.chmod(xstartup_file, 0o755)

    # 4. Kill existing VNC sessions and start fresh
    subprocess.run("vncserver -kill :1", shell=True, capture_output=True)
    subprocess.run("vncserver :1 -geometry 1280x720 -depth 16", shell=True)

    # 5. Download Bore binary if not present
    if not os.path.exists("./bore"):
        subprocess.run("curl -sL https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz | tar -xz", shell=True)
        os.chmod("./bore", 0o755)

    # 6. Launch Bore tunnel in background and capture allocated port
    log_file = "bore.log"
    subprocess.Popen(f"./bore local {LOCAL_VNC_PORT} --to bore.pub > {log_file} 2>&1", shell=True)

    # Wait up to 10 seconds for bore to assign a public port
    public_address = None
    for _ in range(10):
        time.sleep(1)
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                match = re.search(r"listening at (bore\.pub:\d+)", content)
                if match:
                    public_address = match.group(1)
                    break

    return public_address

# Run Setup
with st.spinner("Configuring XFCE desktop components & Bore tunnel..."):
    bore_addr = setup_vnc_and_bore()

if bore_addr:
    st.success("✅ VNC Tunnel Connected Successfully!")
    
    st.subheader("🔑 VNC Connection Details")
    st.code(f"Address: {bore_addr}\nPassword: {VNC_PASS}", language="text")
    st.info(f"👉 In your VNC Viewer app, enter **`{bore_addr}`** and use password **`{VNC_PASS}`**.")
else:
    st.warning("⚠️ VNC is running, but bore.pub is assigning a port. Please refresh the page in 5 seconds.")

st.markdown("---")
st.write("🟢 Server status: **Active 24/7 via NotRa Uptime Dashboard**")
