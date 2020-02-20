import platform
from externalLibs import *
class buildConfig(object):
    def __init__(self,isDebug):
        self.lib = Libs()
        self.incDirs=[self.lib.boost,self.lib.eigen]
        self.linkDir=[self.lib.boost + "/stage/lib"]
        self.linkOpt=["pthread" ,"m","dl"]

        self.CXX="g++"
        self.CCFLAGS=['-std=c++17', '-Wall',"-fpermissive"]
        self.mklroot=self.lib.mkl

        self.isDebug=int(isDebug)
        if(self.isDebug==1):
            self.preDifines=['-DDEBUG']
            self.CCFLAGS.append('-g')
        else:
            self.preDifines=['-DNDEBUG']
            self.CCFLAGS.append('-O3')

class buildConfigForC(buildConfig):
    def __init__(self,isDebug):
        buildConfig.__init__(self,isDebug)
        self.CC="gcc"
        self.CCFLAGS=['-std=c99','-Wall']
        self.isDebug=int(isDebug)
        if(self.isDebug==1):
            self.preDifines=['-DDEBUG']
            self.CCFLAGS.append('-g')
        else:
            self.preDifines=['-DNDEBUG']
            self.CCFLAGS.append('-O3')

class MlpBuildConfig(buildConfig):    
    def __init__(self,isDebug):
        
        buildConfig.__init__(self,isDebug)
        if(self.isDebug==1):
            self.targetName="mlpDebug"
        else:
            self.targetName="mlpRelease"
        distroname = platform.linux_distribution()
        if(distroname[0]=="manjaro"):
            self.incDirs.append("/usr/include/opencv4")
        self.incDirs.append(self.mklroot+"/include")
        self.linkDir.append(self.mklroot+"/lib/intel64")
        self.linkOpt.append(["mkl_intel_lp64","mkl_sequential" ,"mkl_core"])
        self.preDifines.append("EIGEN_USE_MKL_ALL")



