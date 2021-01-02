from paramiko import SSHClient
import paramiko
 
class SSH:
    def __init__(self,host,user,pwd):
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=host,username=user,password=pwd)
 
    def exec_cmd(self,cmd):
        stdin,stdout,stderr = self.ssh.exec_command(cmd)
        if stderr.channel.recv_exit_status() != 0:
            print(stderr.read())
        else:
            print( stdout.read())
# propiltiouracila
if __name__ == '__main__':
    ssh = SSH('200.239.101.201','helber','srcbk,smn9')
    ssh.exec_cmd("pwd")
    ssh.exec_cmd("ls")
    ssh.exec_cmd("echo 'fimm'")
