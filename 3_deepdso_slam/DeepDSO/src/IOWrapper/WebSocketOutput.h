#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include <libwebsockets.h> // Define 'lws_callback_reasons'
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <condition_variable> // Para la sincronización
#include "json.hpp" // nlohmann/json

// Fwd declare
struct lws_context;
struct lws;

using json = nlohmann::json;

namespace dso {
namespace IOWrap {

class WebSocketOutput : public Output3DWrapper
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WebSocketOutput(int port = 9090);
    virtual ~WebSocketOutput();

    // Estas son las funciones que FullSystem llamará
    virtual void publishKeyframes(const std::vector<FrameHessian*>& keyframes, bool final, CalibHessian* Hcalib);
    virtual void publishCamPose(const SE3& camToWorld, CalibHessian* Hcalib); // (Usa SE3)
    virtual void publishGraph(const std::map<long,int>& connectivity);
    virtual void publishTrackedPoints(const std::vector<Vec2>&, const std::vector<float>&, const std::vector<int>&);
    virtual void publishGroundTruth(const Sim3&, double);
    virtual void join(); // Para cerrar el hilo limpiamente

    // ======== AÑADIDO: Función de espera ========
    /**
     * @brief Bloquea el hilo que lo llama hasta que un cliente WebSocket se conecte.
     */
    void waitForClient();
    // ======== FIN DE AÑADIDO ========
    
    // ======== MODIFICADO: Mover broadcast() a public ========
    /**
     * @brief Envía un mensaje a todos los clientes (Thread-Safe).
     */
    void broadcast(const std::string& msg);
    // ======== FIN DE MODIFICADO ========

private:
    // --- Lógica del Servidor ---
    void run(); // El bucle principal del servidor en su hilo
    void stop();
    // void broadcast(const std::string& msg); // <-- Borrado de private
    void processMessageQueue();

    // Callback estático de libwebsockets
    static int lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                            void *user, void *in, size_t len);

    int m_port;
    bool m_running;
    struct lws_context* m_context;
    std::thread m_serverThread;
    std::set<struct lws*> m_clients;

    // Cola de mensajes segura para hilos
    std::queue<std::string> m_messageQueue;
    std::mutex m_queueMutex;

    // Sincronización para el arranque del contexto
    std::mutex m_contextMutex;
    std::condition_variable m_contextReadyCV;
    bool m_contextReady = false;

    // ======== AÑADIDO: Sincronización para la conexión del cliente ========
    std::mutex m_clientMutex;
    std::condition_variable m_clientConnectedCV;
    bool m_clientConnected = false;
    // ======== FIN DE AÑADIDO ========
};

} // namespace IOWrap
} // namespace dso