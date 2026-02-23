#include "WebSocketOutput.h"
#include "FullSystem/FullSystem.h"
#include <libwebsockets.h>
#include <signal.h>
#include <chrono>
#include <cmath>

namespace dso {
namespace IOWrap {

// --- Estructuras para LWS ---
struct per_session_data {
    // Podríamos poner datos por sesión aquí si quisiéramos
};

// --- Implementación de WebSocketOutput ---

WebSocketOutput::WebSocketOutput(int port) : 
    m_port(port), m_running(false), m_context(nullptr) 
{
    printf("INFO: Iniciando WebSocketOutput en puerto %d...\n", m_port);
    m_running = true;
    m_serverThread = std::thread(&WebSocketOutput::run, this);
}

WebSocketOutput::~WebSocketOutput() 
{
    stop();
}

void WebSocketOutput::join() 
{
    if (m_serverThread.joinable()) {
        m_serverThread.join();
    }
}

void WebSocketOutput::stop() 
{
    if (!m_running) return;
    m_running = false;

    if (m_context) {
        lws_cancel_service(m_context);
    }

    join();

    if (m_context) {
        lws_context_destroy(m_context);
        m_context = nullptr;
    }
    printf("INFO: WebSocketOutput detenido.\n");
}

void WebSocketOutput::waitForClient()
{
    printf("INFO: Esperando la conexión del cliente de Android...\n");
    {
        std::unique_lock<std::mutex> lock(m_clientMutex);
        m_clientConnectedCV.wait(lock, [this]{ return m_clientConnected; });
    }
    printf("INFO: ¡Cliente conectado! Iniciando el SLAM...\n");
}

// Bucle principal del servidor
void WebSocketOutput::run() 
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa, 0);

    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));

    struct lws_protocols protocols[2] = {
        // Tu búfer de 1MB está perfecto
        { "dso-protocol", &WebSocketOutput::lws_callback, sizeof(per_session_data), 1024 * 1024, 0, NULL, 0 }, 
        { NULL, NULL, 0, 0, 0, NULL, 0 }
    };

    info.port = m_port;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;
    info.options = LWS_SERVER_OPTION_VALIDATE_UTF8;
    info.user = this;

    m_context = lws_create_context(&info);

    {
        std::lock_guard<std::mutex> lock(m_contextMutex);
        m_contextReady = true;
    }
    m_contextReadyCV.notify_all();

    if (!m_context) {
        fprintf(stderr, "ERROR: Creación del contexto de lws fallida.\n");
        m_running = false;
        return;
    }

    // Eliminamos el "Modo Simulación". Solo procesamos la cola.
    while (m_running) {
        if (lws_service(m_context, 50) < 0) {
             m_running = false;
        }
        processMessageQueue();
    }
}

// Envía un mensaje a la cola (Thread-Safe)
void WebSocketOutput::broadcast(const std::string& msg) 
{
    {
        std::unique_lock<std::mutex> lock(m_contextMutex);
        m_contextReadyCV.wait(lock, [this]{ return m_contextReady; });
    }

    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_messageQueue.push(msg); // Solo encola el JSON crudo
    if(m_context) {
        lws_cancel_service(m_context);
    }
}

// Procesa la cola de mensajes (llamado desde el hilo de lws)
void WebSocketOutput::processMessageQueue() 
{
    std::lock_guard<std::mutex> lock(m_queueMutex);

    if (!m_messageQueue.empty()) {
        printf("DEBUG: Procesando %ld mensajes en la cola.\n", m_messageQueue.size());
        fflush(stdout);
    }

    while (!m_messageQueue.empty()) {
        std::string msg = m_messageQueue.front();
        m_messageQueue.pop();

        // ======== AÑADIDO: Newline Separator (Punto Único de Salida) ========
        // Añadimos el \n aquí. Es el único lugar donde se escriben datos.
        std::string msgWithNewline = msg + "\n";
        // ======== FIN DE AÑADIDO ========

        unsigned char buffer[LWS_PRE + msgWithNewline.length()];
        memcpy(&buffer[LWS_PRE], msgWithNewline.c_str(), msgWithNewline.length());

        for (auto wsi : m_clients) {
            int bytes_sent = lws_write(wsi, &buffer[LWS_PRE], msgWithNewline.length(), LWS_WRITE_TEXT);

            if (bytes_sent == (int)msgWithNewline.length()) {
                printf("DEBUG: Enviados %d bytes al cliente (WSI: %p).\n", bytes_sent, (void*)wsi);
                fflush(stdout);
            } else if (bytes_sent >= 0) {
                fprintf(stderr, "ERROR: WebSocket lws_write falló (ENVÍO PARCIAL). Enviados %d de %ld bytes.\n", bytes_sent, msgWithNewline.length());
                fflush(stderr);
            } else {
                fprintf(stderr, "ERROR: WebSocket lws_write falló (código de error %d).\n", bytes_sent);
                fflush(stderr);
            }
        }
    }
}

// --- Implementación de los Callbacks de Output3DWrapper ---

// ======== MODIFICADO: SILENCIAR ESTAS FUNCIONES ========
void WebSocketOutput::publishCamPose(const SE3& camToWorld, CalibHessian* Hcalib) 
{
    // NO HACER NADA.
    // El 'transmitterThread' en main_dso_pangolin.cpp se encarga de esto.
}

void WebSocketOutput::publishKeyframes(const std::vector<FrameHessian*>& keyframes, bool final, CalibHessian* Hcalib) 
{
    // NO HACER NADA.
    // El 'transmitterThread' en main_dso_pangolin.cpp se encarga de esto.
}
// ======== FIN DE LA MODIFICACIÓN ========


void WebSocketOutput::publishGraph(const std::map<long,int>& connectivity) {}
void WebSocketOutput::publishTrackedPoints(const std::vector<Vec2>&, const std::vector<float>&, const std::vector<int>&) {}
void WebSocketOutput::publishGroundTruth(const Sim3&, double) {}

int WebSocketOutput::lws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                                void *user, void *in, size_t len) 
{
    struct lws_context *context = lws_get_context(wsi);
    WebSocketOutput *server = (WebSocketOutput *)lws_context_user(context);

    if (!server) return -1;

    switch (reason) {
        case LWS_CALLBACK_ESTABLISHED:
        { 
            printf("INFO: Cliente conectado (WSI: %p)\n", (void*)wsi);
            server->m_clients.insert(wsi);

            // ======== CORREGIDO: Bug de Doble Conexión ========
            {
                std::lock_guard<std::mutex> lock(server->m_clientMutex);
                server->m_clientConnected = true;
            }
            server->m_clientConnectedCV.notify_all(); // ¡Despertar a TODOS los hilos!
            // ======== FIN DE CORRECCIÓN ========

            json payload;
            payload["type"] = "CONNECTION_ESTABLISHED";
            payload["message"] = "Conexión exitosa al servidor DSO.";

            // Usar 'broadcast' para encolar el mensaje de "hola".
            // Así mantenemos un único punto de escritura (processMessageQueue)
            server->broadcast(payload.dump());

            break;
        } 

        case LWS_CALLBACK_CLOSED:
            printf("INFO: Cliente desconectado (WSI: %p)\n", (void*)wsi);
            server->m_clients.erase(wsi);
            break;
        default:
            break;
    }
    return 0;
}

} // namespace IOWrap
} // namespace dso