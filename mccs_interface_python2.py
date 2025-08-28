import subprocess
import os
import re
import json
import shutil

DEBUG_ENABLED = False

################################################################
# class Ccf
################################################################

class Ccf(object):
    # never construct a Ccf object by hand!
    # always use the Host.pull() method to retrieve an instance of this class.
    def __init__(self, configuration, instance, version, content):
        self.configuration = configuration
        self.instance      = instance
        self.version       = version
        self.content       = content

    def read_only(self):
        return self.content

################################################################
# class Host
################################################################

def execute(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return { "return_code": process.returncode, "output": stdout.decode() + stderr.decode() }

class Host(object):

    def __init__(self):
        self.tmp_dir = os.path.join("/tmp", "mccs_tmp__" + str(os.getpid()))

        # remove leftover temporary data
        # from previous services
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        # create this hosts tmp dir
        os.makedirs(self.tmp_dir)

    def ls(self):
        try:
            result = execute("mccs ls -j")

            if DEBUG_ENABLED:
                print("----------------------------------------------")
                print("command     = ls")
                print("exit status =", result["return_code"])
                print("stdout      =", result["output"])
                print("----------------------------------------------\n\n")

            return json.loads(result["output"], strict=False)

        except Exception as err:
            print("error:", err)
            print("result:", result)
            exit(1)
    
    def fmt(self, ccf_path):
        try:
            result = execute("mccs fmt " + ccf_path)

            if DEBUG_ENABLED:
                print("----------------------------------------------")
                print("command     = fmt")
                print("exit status =", result["return_code"])
                print("stdout      =", result["output"])
                print("----------------------------------------------\n\n")
        
        except Exception as err:
            print("error:", err)
            print("result:", result)
            exit(1)

    def pull(self, configuration, instance, version = -1):
        ccf_fname = configuration + "___" + instance + ".ccf"
        ccf_path  = os.path.join(self.tmp_dir, ccf_fname)
        
        if os.path.exists(ccf_path):
            os.remove(ccf_path)

        result = None
        try:
            if version == -1:
                result = execute("cd " + self.tmp_dir + " && mccs pull " + configuration + " " + instance)
            else:
                result = execute("cd " + self.tmp_dir + " && mccs pull " + configuration + " " + instance + " " + str(version))

            if DEBUG_ENABLED:
                print("----------------------------------------------")
                print("command     = pull")
                print("exit status =", result["return_code"])
                print("stdout      =", result["output"])
                print("----------------------------------------------\n\n")

        except Exception as err:
            print("error:", err)
            print("result:", result)
            exit(1)

        with open(ccf_path, mode="r") as f: 
            ccf_content = f.read()

        ccf = Ccf(configuration, instance, version, json.loads(ccf_2_json(ccf_content)))
        os.remove(ccf_path)

        return ccf

    def push(self, ccf, message):
        assert ccf.configuration is not None
        assert ccf.instance      is not None

        # steps: pre-pull, rename & merge are neccessary to preserve comments

        #-------------------------------------------------------------
        # pre pull & rename
        #-------------------------------------------------------------

        ccf_fname = ccf.configuration + "___" + ccf.instance + ".ccf"
        ccf_path  = os.path.join(self.tmp_dir, ccf_fname)
        ccf_tmp   = os.path.join(self.tmp_dir, "tmp.ccf")
        
        if os.path.exists(ccf_path):
            os.remove(ccf_path)

        try:
            result = execute("cd " + self.tmp_dir + " && mccs pull " + " " + ccf.configuration + " " + ccf.instance)
            
            if DEBUG_ENABLED:
                print("----------------------------------------------")
                print("command     = pull")
                print("exit status =", result["return_code"])
                print("stdout      =", result["output"])
                print("----------------------------------------------\n\n")

        except Exception as err:
            print("error:", err)
            print("result:", result)
            exit(1)

        os.rename(ccf_path, ccf_tmp)

        #-------------------------------------------------------------
        # generate ccf from ccf object
        #-------------------------------------------------------------
        
        ccf_content = json_2_ccf(ccf.content)

        with open(ccf_path, "w") as f:
            f.write(ccf_content)

        self.fmt(ccf_path)

        #-------------------------------------------------------------
        # merge pre pulled ccf with generated ccf
        #-------------------------------------------------------------

        try:
            result = execute("mccs merge ccf " + ccf_tmp + " " + ccf_path)
            
            if DEBUG_ENABLED:
                print("----------------------------------------------")
                print("command     = merge")
                print("exit status =", result["return_code"])
                print("stdout      =", result["output"])
                print("----------------------------------------------\n\n")

        except Exception as err:
            print("error:", err)
            print("result:", result)
            exit(1)


        #-------------------------------------------------------------
        # push
        #-------------------------------------------------------------

        try:
            result = execute("mccs push " + ccf_path + " '" + message + "'")

            if DEBUG_ENABLED:
                print("----------------------------------------------")
                print("command     = push")
                print("exit status =", result["return_code"])
                print("stdout      =", result["output"])
                print("----------------------------------------------\n\n")

        except Exception as err:
            print("error:", err)
            print("result:", result)
            exit(1)
        
        os.remove(ccf_fname)
        os.remove(ccf_tmp)

    def get_configurations(self):
        data           = self.ls()
        configurations = []

        for configuration in data["data"]:
            configurations.append(configuration["configuration_name"])

        return configurations

    def get_instances(self, configuration_name):
        data         = self.ls()
        instances    = []
        found_config = False

        for configuration in data["data"]:
            if configuration["configuration_name"] == configuration_name:
                found_config = True

                for instance in configuration["instances"]:
                    instances.append(instance["instance_name"])

        if found_config:
            return instances
        else:
            return None
        
    def get_versions(self, configuration_name, instance_name):
        data         = self.ls()
        versions     = []
        found_config = False
        found_inst   = False

        for configuration in data["data"]:
            if configuration["configuration_name"] == configuration_name:
                found_config = True

                for instance in configuration["instances"]:
                    if instance["instance_name"] == instance_name:
                        found_inst = True

                        for version in instance["versions"]:
                            try:
                                version_id = int(version["version_id"])
                                versions.append(version_id)
                            except:
                                print("----------------------------------------------")
                                print("------------------ABORT-----------------------")
                                print("----------------------------------------------")
                                print("invalid version-id '" + version["version_id"] + "'")
                                exit(1)

        if found_config and found_inst:
            return versions
        else:
            return None
        
    def has_configuration(self, configuration_name):
        configurations = self.get_configurations()
        return configuration_name in configurations
    
    def has_instance(self, configuration_name, instance_name):
        if not self.has_configuration(configuration_name):
            return False

        instances = self.get_instances(configuration_name)
        return instance_name in instances
    
    def has_version(self, configuration_name, instance_name, version):
        if not self.has_instance(configuration_name, instance_name):
            return False

        versions = self.get_versions(configuration_name, instance_name)
        return version in versions

################################################################
# Conversions
################################################################

def ccf_2_json(ccf_content):
    
    # remove all comment lines, to make it valid json
    ccf_content = re.sub("#.*\n", "", ccf_content)
    
    # search all enum values a::b and replace them with "enum::a::b"
    already_replaced = {}
    search_results   = re.findall('[A-Za-z_][A-Za-z_0-9]+::[A-Za-z_][A-Za-z_0-9]+', ccf_content)
    for result in search_results:
        if result in already_replaced:
            continue

        new_val = '"' + "enum::" + result + '"'
        ccf_content = ccf_content.replace(result, new_val)
        already_replaced[result] = True


    # search all properties x and replace them with "x"
    search_results = re.findall(".[A-Za-z_][A-Za-z0-9_]*\s*::?", ccf_content)
    search_results = sorted(search_results, key=len)
    search_results.reverse()

    for result in search_results:
        if len(result) > 2 and result[-2] == ":" and result[-1] == ":":
            continue

        new_val = '"' + result[0:-1].strip() + '":'
        ccf_content = ccf_content.replace(result, new_val)

    ccf_content = re.sub(r'\\([a-zA-Z])', r'\\\\\1', ccf_content)
    return ccf_content

def json_2_ccf(json_content):
    
    json_content = json.dumps(json_content)

    # search for all patterns "enum::a::b" and replace them with a::b
    search_results = re.findall('"enum::[A-Za-z_][A-Za-z_0-9]*::[A-Za-z_][A-Za-z_0-9]*"', json_content)
    for result in search_results:
        new_val      = result[7:-1]
        json_content = json_content.replace(result, new_val)

    # search for all patterns "x": and replace them with x:
    search_results = re.findall('"[A-Za-z_][A-Za-z_0-9]*"\s*:', json_content)
    for result in search_results:
        new_val      = result.replace('"', '')
        json_content = json_content.replace(result, new_val)

    return json_content
