import pdb,sys,bdb
import os
import remote_pdb
import errno
import logging
import os
import re
import time
import socket
import sys
import threading
import subprocess
from pdb import Pdb

server_socket = None

class RemoteNvitop():

    def __init__(self, host, port):
        global server_socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)  # 最大连接数
        self.server_socket = server_socket
        print(f" \033[91m remote nvitop listening on {host}:{port} \033[0m")
        client_handler = threading.Thread(target=self.handle_client_connection,)
        client_handler.start()

    def handle_client_connection(self,):
        client_socket, addr = self.server_socket.accept()
        print("Remote accepted connection from %s." % repr(addr))
        try:
            while True:
                time.sleep(1)
                # 运行nvitop命令
                process = subprocess.Popen(['nvitop'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # process = subprocess.Popen(['script', '-q', '-c', 'nvitop', '/dev/null'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # 读取nvitop的输出并通过TCP发送
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # 发送清屏命令
                        # client_socket.sendall(b'\033[2J\033[H')
                        client_socket.sendall(output.encode('utf-8'))
        finally:
            client_socket.close()

class RemotePdb(remote_pdb.RemotePdb):
    """
    This will run pdb as an ephemeral telnet service. Once you connect, no one
    else can connect. On construction, this object will block execution till a
    client has connected.

    Based on https://github.com/tamentis/rpdb I think ...

    To use this::

        RemotePdb(host='0.0.0.0', port=4444).set_trace()

    Then run: telnet 127.0.0.1 4444
    """
    active_instance = None

    def __init__(self, host, port, patch_stdstreams=False, quiet=False):
        self._quiet = quiet
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.listen_socket.bind((host, port))
        if not self._quiet:
            print(" \033[91m  RemotePdb session open at %s:%s, waiting for connection ... \033[0m" % self.listen_socket.getsockname())
        self.listen_socket.listen(1)
        self.connection = None
        self.address = None
        self.handle = None

        self._start_listener_thread()

        self.backup = []
        if patch_stdstreams:
            for name in (
                    'stderr',
                    'stdout',
                    '__stderr__',
                    '__stdout__',
                    'stdin',
                    '__stdin__',
            ):
                self.backup.append((name, getattr(sys, name)))
                setattr(sys, name, self.handle)
        RemotePdb.active_instance = self

    def _start_listener_thread(self):
        listener_thread = threading.Thread(target=self._accept_connection)
        listener_thread.daemon = True
        listener_thread.start()

    def _accept_connection(self):
        self.connection, self.address = self.listen_socket.accept()
        if not self._quiet:
            print("RemotePdb accepted connection from %s." % repr(self.address))
        self.handle = remote_pdb.LF2CRLF_FileWrapper(self.connection)
        Pdb.__init__(self, completekey='tab', stdin=self.handle, stdout=self.handle)

    def valid(self, info=False):
        if self.handle is None:
            print(" \033[91m RemotePdb is not connected yet. ", "RemotePdb session open at %s:%s, waiting for connection ... \033[0m" % self.listen_socket.getsockname())
            while self.handle is None:
                if info:
                    print(" \033[91m RemotePdb is not connected yet. ", "RemotePdb session open at %s:%s, waiting for connection ... \033[0m" % self.listen_socket.getsockname())
                time.sleep(5)

    def set_trace(self, frame=None):
        self.valid()
        if frame is None:
            frame = sys._getframe().f_back
        try:
            pdb.Pdb.set_trace(self, frame)
        except IOError as exc:
            if exc.errno != errno.ECONNRESET:
                raise

    def post_mortem(self, t=None):
        self.valid(info=True)
        # handling the default
        if t is None:
            # sys.exc_info() returns (type, value, traceback) if an exception is
            # being handled, otherwise it returns None
            t = sys.exc_info()[2]
        if t is None:
            raise ValueError("A valid traceback must be passed if no "
                            "exception is being handled")
        
        self.reset()
        self.interaction(None, t)


def find_available_port(start_port, step=1):
    port = start_port
    while True:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind(('0.0.0.0', port))
            test_socket.close()
            return port
        except OSError:
            port += step

import fcntl 

def setup_remote_pdb(local_rank):
    lockfile = '/tmp/pdb.lock'
    lock = open(lockfile, 'w')
    try:
        # Try to acquire the lock
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # Check if the environment variable is already set
        if not os.getenv(f'pdb{local_rank}'):
            # Find an available port
            port = find_available_port(20000)
            
            # Create a RemotePdb instance
            _pdb = RemotePdb('0.0.0.0', port)
            pdb.set_trace = _pdb.set_trace
            pdb.post_mortem = _pdb.post_mortem
            
            # Set the environment variable
            os.environ[f'pdb{local_rank}'] = os.getenv('LOCAL_RANK', '0')
    except IOError:
        # Another process has acquired the lock, so we just print the existing environment variable
        pass
    finally:
        # Always release the lock
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

# port = find_available_port(10000)
# if int(os.environ.get('LOCAL_RANK', 0)) == 0:
#     RemoteNvitop('0.0.0.0', port)
local_rank = os.getenv('LOCAL_RANK', '0')
setup_remote_pdb(local_rank)

def rank0_debugger():
    if os.getenv('LOCAL_RANK', '0') == '0':
        pdb.set_trace()

pdb.set_trace0 = rank0_debugger
print(os.getenv(f'pdb{local_rank}'))