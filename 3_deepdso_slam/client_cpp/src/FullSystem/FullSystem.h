/**
 * This file is part of DSO.
 * ... (resto de la licencia sin cambios) ...
 */

#pragma once
#define MAX_ACTIVE_FRAMES 100

#include <deque>
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "vector"

#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/PixelSelector2.h"

#include <cmath> // MODIFICADO: Se usa <cmath> en lugar de <math.h> por estándar de C++

// MODIFICADO: Declaración adelantada (forward declaration) de cv::Mat.
// Esto evita tener que incluir todo el header de OpenCV aquí, haciendo la compilación más rápida y limpia.
namespace cv {
    class Mat;
}

namespace dso
{
namespace IOWrap
{
class Output3DWrapper;
}

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
class ImageAndExposure;
class CoarseDistanceMap;

class EnergyFunctional;

// ... (resto de las funciones de utilidad 'deleteOut' sin cambios) ...
template<typename T> inline void deleteOut(std::vector<T*> &v, const int i)
{
    delete v[i];
    v[i] = v.back();
    v.pop_back();
}
template<typename T> inline void deleteOutPt(std::vector<T*> &v, const T* i)
{
    delete i;
    for(unsigned int k=0;k<v.size();k++)
        if(v[k] == i)
        {
            v[k] = v.back();
            v.pop_back();
        }
}
template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const int i)
{
    delete v[i];
    for(unsigned int k=i+1; k<v.size();k++)
        v[k-1] = v[k];
    v.pop_back();
}
template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const T* element)
{
    int i=-1;
    for(unsigned int k=0; k<v.size();k++)
    {
        if(v[k] == element)
        {
            i=k;
            break;
        }
    }
    assert(i!=-1);

    for(unsigned int k=i+1; k<v.size();k++)
        v[k-1] = v[k];
    v.pop_back();

    delete element;
}


inline bool eigenTestNan(const MatXX &m, std::string msg)
{
    bool foundNan = false;
    for(int y=0;y<m.rows();y++)
        for(int x=0;x<m.cols();x++)
        {
            if(!std::isfinite((double)m(y,x))) foundNan = true;
        }

    if(foundNan)
    {
        printf("NAN in %s:\n",msg.c_str());
        std::cout << m << "\n\n";
    }
    return foundNan;
}

class FullSystem {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FullSystem(const std::string &path_cnn);
    virtual ~FullSystem();

    void addActiveFrame(ImageAndExposure* image, int id);
    void marginalizeFrame(FrameHessian* frame);
    void blockUntilMappingIsFinished();
    float optimize(int mnumOptIts);
    void printResult(std::string file);
    void debugPlot(std::string name);
    void printFrameLifetimes();

    std::vector<IOWrap::Output3DWrapper*> outputWrapper;

    bool isLost;
    bool initFailed;
    bool initialized;
    bool linearizeOperation;

    void setGammaFunction(float* BInv);
    void setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH);

    // MODIFICADO: La declaración de la función ahora es correcta gracias a la forward declaration de cv::Mat
    cv::Mat getDepthMap(FrameHessian* fh);

private:
    CalibHessian Hcalib;

    // ... (resto de funciones privadas sin cambios) ...
    int optimizePoint(PointHessian* point, int minObs, bool flagOOB);
    PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);
    double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);
    Vec4 trackNewCoarse(FrameHessian* fh);
    void traceNewCoarse(FrameHessian* fh);
    void activatePoints();
    void activatePointsMT();
    void activatePointsOldFirst();
    void flagPointsForRemoval();
    void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
    void initializeFromInitializer(FrameHessian* newFrame);
    void flagFramesForMarginalization(FrameHessian* newFH);
    
    // MODIFICADO: Comentamos la función relacionada con la arquitectura CNN antigua
    // void initializeFromInitializerCNN(FrameHessian* newFrame);

    void removeOutliers();
    void setPrecalcValues();
    void solveSystem(int iteration, double lambda);
    Vec3 linearizeAll(bool fixLinearization);
    bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
    void backupState(bool backupLastStep);
    void loadSateBackup();
    double calcLEnergy();
    double calcMEnergy();
    void linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
    void activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,std::vector<ImmaturePoint*>* toOptimize,int min, int max, Vec10* stats, int tid);
    void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);
    void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);
    void debugPlotTracking();
    std::vector<VecX> getNullspaces(
            std::vector<VecX> &nullspaces_pose,
            std::vector<VecX> &nullspaces_scale,
            std::vector<VecX> &nullspaces_affA,
            std::vector<VecX> &nullspaces_affB);
    void setNewFrameEnergyTH();
    void printLogLine();
    
    std::ofstream* calibLog;
    std::ofstream* numsLog;
    std::ofstream* errorsLog;
    std::ofstream* eigenAllLog;
    std::ofstream* eigenPLog;
    std::ofstream* eigenALog;
    std::ofstream* DiagonalLog;
    std::ofstream* variancesLog;
    std::ofstream* nullspacesLog;
    std::ofstream* coarseTrackingLog;
    
    long int statistics_lastNumOptIts;
    long int statistics_numDroppedPoints;
    long int statistics_numActivatedPoints;
    long int statistics_numCreatedPoints;
    long int statistics_numForceDroppedResBwd;
    long int statistics_numForceDroppedResFwd;
    long int statistics_numMargResFwd;
    long int statistics_numMargResBwd;
    float statistics_lastFineTrackRMSE;

    boost::mutex trackMutex;
    std::vector<FrameShell*> allFrameHistory;
    CoarseInitializer* coarseInitializer;
    Vec5 lastCoarseRMSE;

    boost::mutex mapMutex;
    std::vector<FrameShell*> allKeyFramesHistory;
    
    EnergyFunctional* ef;
    IndexThreadReduce<Vec10> treadReduce;
    
    float* selectionMap;
    PixelSelector* pixelSelector;
    CoarseDistanceMap* coarseDistanceMap;
    
    std::vector<FrameHessian*> frameHessians;
    std::vector<PointFrameResidual*> activeResiduals;
    float currentMinActDist;
    
    std::vector<float> allResVec;
    
    boost::mutex coarseTrackerSwapMutex;
    CoarseTracker* coarseTracker_forNewKF;
    CoarseTracker* coarseTracker;
    float minIdJetVisTracker, maxIdJetVisTracker;
    float minIdJetVisDebug, maxIdJetVisDebug;
    
    boost::mutex shellPoseMutex;
    
    void makeKeyFrame( FrameHessian* fh);
    void makeNonKeyFrame( FrameHessian* fh);
    void deliverTrackedFrame(FrameHessian* fh, bool needKF);
    void mappingLoop();
    
    boost::mutex trackMapSyncMutex;
    boost::condition_variable trackedFrameSignal;
    boost::condition_variable mappedFrameSignal;
    std::deque<FrameHessian*> unmappedTrackedFrames;
    int needNewKFAfter;
    boost::thread mappingThread;
    bool runMapping;
    bool needToKetchupMapping;
    int lastRefStopID;
};
}