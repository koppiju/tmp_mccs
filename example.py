# import mccs_interface_python2 as mccs_interface

from mccs_interface_python2 import Host as MccsInterface


ccf = MccsInterface().pull("station", "rsc1").read_only()
print(ccf)

