#include <GLFW/glfw3.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

static int W = 1280;
static int H = 720;

static unsigned int s32 = 0x12345u;
static inline unsigned int xorshift() { s32 ^= s32 << 13; s32 ^= s32 >> 17; s32 ^= s32 << 5; return s32; }
static inline float frand() { return (xorshift() & 0xFFFFFF) / 16777215.0f; }
static inline float hash11(int x) { unsigned int h = (unsigned int)x * 0x27d4eb2d; h ^= h >> 15; return (h & 0xFFFFFF) / 16777215.0f; }

struct Vec3 { float x, y, z; };
static inline Vec3 v3(float x, float y, float z) { return { x,y,z }; }
static inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x,a.y + b.y,a.z + b.z }; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x,a.y - b.y,a.z - b.z }; }
static inline Vec3 operator*(Vec3 a, float s) { return { a.x * s,a.y * s,a.z * s }; }
static inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(Vec3 a, Vec3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline float len(Vec3 a) { return std::sqrt(dot(a, a)); }
static inline Vec3 norm(Vec3 a) { float L = len(a); return L > 1e-6f ? a * (1.0f / L) : v3(0, 0, 0); }

static const int   COLS = 24;
static const int   ROWS = 16;
static const float CELL = 2.2f;  
static const float HGT = 1.6f;  

struct Cell { bool visited = false; bool wall[4] = { true,true,true,true }; };
static std::vector<Cell> grid;
static inline int idx(int c, int r) { return r * COLS + c; }

static void generateMaze() {
    grid.assign(COLS * ROWS, Cell{});
    std::vector<int> st; st.reserve(COLS * ROWS);
    int c = xorshift() % COLS, r = xorshift() % ROWS; grid[idx(c, r)].visited = true; st.push_back(idx(c, r));
    while (!st.empty()) {
        int i = st.back(); int cc = i % COLS, rr = i / COLS; int dirs[4] = { 0,1,2,3 };
        for (int k = 3; k > 0; --k) { int j = xorshift() % (k + 1); std::swap(dirs[k], dirs[j]); }
        bool adv = false; for (int d = 0; d < 4; ++d) { int dir = dirs[d]; int nc = cc, nr = rr; if (dir == 0)nr--; else if (dir == 1)nc++; else if (dir == 2)nr++; else nc--; if (nc < 0 || nc >= COLS || nr < 0 || nr >= ROWS) continue; if (!grid[idx(nc, nr)].visited) { grid[idx(cc, rr)].wall[dir] = false; grid[idx(nc, nr)].wall[(dir + 2) % 4] = false; grid[idx(nc, nr)].visited = true; st.push_back(idx(nc, nr)); adv = true; break; } }
        if (!adv) st.pop_back();
    }
}

static Vec3 camPos; static float yaw = 0.0f, pitch = 0.0f; static bool autoFly = true;
static double mx = 0, my = 0; static bool dragging = false;

static void updateAutoFly(float dt) {
    float cx = COLS * CELL * 0.5f, cz = ROWS * CELL * 0.5f; float rad = std::min(COLS, ROWS) * CELL * 0.22f;
    float spd = 0.2f; float t = (float)glfwGetTime() * spd;
    camPos.x = cx + rad * std::sin(t);
    camPos.z = cz + rad * std::sin(t) * std::cos(t);
    camPos.y = 0.5f + 0.15f * std::sin(t * 0.9f);
    float tx = cx, tz = cz; Vec3 dir = norm(v3(tx - camPos.x, 0.1f, tz - camPos.z));
    yaw = std::atan2(dir.x, dir.z); pitch = -0.02f;
}

static Vec3 forward() { return v3(std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch)); }
static Vec3 rightv() { Vec3 f = forward(); return v3(f.z, 0, -f.x); }

static void setPerspective(float fovY, float zNear, float zFar) {
    float aspect = (float)W / (float)H; float f = 1.0f / std::tan(fovY * 0.5f * (3.14159265f / 180.0f));
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    float M[16] = { f / aspect,0,0,0, 0,f,0,0, 0,0,(zFar + zNear) / (zNear - zFar),-1, 0,0,(2 * zFar * zNear) / (zNear - zFar),0 };
    glLoadMatrixf(M); glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

static void lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = norm(center - eye); Vec3 s = norm(cross(f, up)); Vec3 u = cross(s, f);
    float M[16] = { s.x, u.x, -f.x, 0, s.y, u.y, -f.y, 0, s.z, u.z, -f.z, 0, 0, 0, 0, 1 }; glMultMatrixf(M);
    glTranslatef(-eye.x, -eye.y, -eye.z);
}

static GLuint texBrick = 0, texTile = 0, texCeil = 0; static int texSize = 64;

static bool  DISP_ON = true;         
static float DISP_SCALE = 0.06f;
static void makeTexture(GLuint& tex, int Wt, int Ht, const std::vector<unsigned char>& rgba) {
    if (!tex) glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Wt, Ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
}

static void genBrickTex(int N) {
    int Wt = N, Ht = N; std::vector<unsigned char> rgba(Wt * Ht * 4);
    for (int y = 0; y < Ht; y++) {
        for (int x = 0; x < Wt; x++) {
            float ny = (float)y / Ht; float nx = (float)x / Wt;
            int row = (int)(ny * 8.0f); float off = (row % 2) ? 0.5f : 0.0f; float fx = std::fmod(nx + off, 1.0f);
            int col = (int)(fx * 8.0f);
            
            bool mortar = ((y % 8) == 0) || ((x + ((row % 2) ? Wt / 16 : 0)) % 8 == 0);
            unsigned char r, g, b;
            if (mortar) { r = 235; g = 235; b = 235; }
            else {
                
                int baseR = 170 + (int)(hash11(row * 31 + col) * 40);
                int baseG = 30 + (int)(hash11(row * 17 + col) * 20);
                int baseB = 28 + (int)(hash11(row * 13 + col) * 20);
                
                baseR += (int)(std::sin((x ^ y) * 0.17f) * 6.0f);
                r = (unsigned char)std::min(255, std::max(0, baseR)); g = (unsigned char)baseG; b = (unsigned char)baseB;
            }
            int i = (y * Wt + x) * 4; rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = 255;
        }
    }
    makeTexture(texBrick, Wt, Ht, rgba);
}

static void genTileTex(int N) {
    int Wt = N, Ht = N; std::vector<unsigned char> rgba(Wt * Ht * 4);
    for (int y = 0; y < Ht; y++) {
        for (int x = 0; x < Wt; x++) {
            bool grout = (x % 8 == 0) || (y % 8 == 0);
            unsigned char r = (grout ? 200 : 170 + (unsigned char)((((x * 13) ^ (y * 7)) & 15)));
            unsigned char g = (grout ? 200 : 170 + (unsigned char)((((x * 11) ^ (y * 5)) & 15)));
            unsigned char b = (grout ? 200 : 170 + (unsigned char)((((x * 9) ^ (y * 3)) & 15)));
            int i = (y * Wt + x) * 4; rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = 255;
        }
    }
    makeTexture(texTile, Wt, Ht, rgba);
}

static void genCeilTex(int N) {
    int Wt = N, Ht = N; std::vector<unsigned char> rgba(Wt * Ht * 4);
    for (int y = 0; y < Ht; y++) {
        for (int x = 0; x < Wt; x++) {
            unsigned char r = 210 + ((x * y) & 3); unsigned char g = 170 + ((x + 2 * y) & 3); unsigned char b = 20 + ((x + y) & 3);
            int i = (y * Wt + x) * 4; rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = 255;
        }
    }
    makeTexture(texCeil, Wt, Ht, rgba);
}

static void regenTextures() { genBrickTex(texSize); genTileTex(texSize); genCeilTex(texSize); }

static void drawTexturedQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, float u0, float v0, float u1, float v1) {
    glBegin(GL_QUADS);
    glTexCoord2f(u0, v0); glVertex3f(a.x, a.y, a.z);
    glTexCoord2f(u1, v0); glVertex3f(b.x, b.y, b.z);
    glTexCoord2f(u1, v1); glVertex3f(c.x, c.y, c.z);
    glTexCoord2f(u0, v1); glVertex3f(d.x, d.y, d.z);
    glEnd();
}

static inline float brickHeight(float u, float v)
{
    float U = u - std::floor(u);
    float V = v - std::floor(v);

    int row = (int)(V * 8.0f);
    float off = (row % 2) ? 0.5f : 0.0f;       
    float fx = std::fmod(U + off, 1.0f);
    int col = (int)(fx * 8.0f);

    bool mortarH = (((int)(V * 64.0f)) % 8) == 0;
    bool mortarV = (((int)((U * 64.0f) + ((row % 2) ? 4 : 0))) % 8) == 0;
    bool mortar = mortarH || mortarV;

    float base = mortar ? 0.0f : 1.0f;

    float n = hash11(row * 31 + col) * 0.25f;

    float du = std::fmod((U + off) * 8.0f, 1.0f);
    du = std::min(du, 1.0f - du);
    float dv = std::fmod(V * 8.0f, 1.0f);
    dv = std::min(dv, 1.0f - dv);
    float bevel = std::min(du, dv);               
    float bevelFactor = 1.0f - 4.0f * std::min(bevel, 0.25f); 
    float h = base * (0.6f + 0.4f * bevelFactor) + n;

    if (h < 0.0f) h = 0.0f;
    if (h > 1.0f) h = 1.0f;
    return h;
}

static void drawWall(Vec3 a, Vec3 b)
{
    Vec3 t = b - a;
    Vec3 up = v3(0, 1, 0);
    Vec3 n = norm(cross(up, t));      
    glBindTexture(GL_TEXTURE_2D, texBrick);

    float lenW = len(t);

    const int SX = 32;   
    const int SY = 16;   

    const float uMax = lenW * 0.8f;  
    const float vMax = HGT * 0.8f;

    for (int iy = 0; iy < SY; ++iy)
    {
        for (int ix = 0; ix < SX; ++ix)
        {
           
            float tx0 = (float)ix / (float)SX;
            float tx1 = (float)(ix + 1) / (float)SX;
            float ty0 = (float)iy / (float)SY;
            float ty1 = (float)(iy + 1) / (float)SY;

            Vec3 p0 = a + t * tx0 + v3(0, HGT * ty0, 0);
            Vec3 p1 = a + t * tx1 + v3(0, HGT * ty0, 0);
            Vec3 p2 = a + t * tx1 + v3(0, HGT * ty1, 0);
            Vec3 p3 = a + t * tx0 + v3(0, HGT * ty1, 0);

            float u0 = uMax * tx0;
            float u1 = uMax * tx1;
            float v0 = vMax * ty0;
            float v1 = vMax * ty1;

            if (DISP_ON && DISP_SCALE > 0.0f)
            {
               
                float h0 = brickHeight(u0, v0);
                float h1 = brickHeight(u1, v0);
                float h2 = brickHeight(u1, v1);
                float h3 = brickHeight(u0, v1);

                p0 = p0 + n * (h0 * DISP_SCALE);
                p1 = p1 + n * (h1 * DISP_SCALE);
                p2 = p2 + n * (h2 * DISP_SCALE);
                p3 = p3 + n * (h3 * DISP_SCALE);
            }

            glBegin(GL_QUADS);
            glTexCoord2f(u0, v0); glVertex3f(p0.x, p0.y, p0.z);
            glTexCoord2f(u1, v0); glVertex3f(p1.x, p1.y, p1.z);
            glTexCoord2f(u1, v1); glVertex3f(p2.x, p2.y, p2.z);
            glTexCoord2f(u0, v1); glVertex3f(p3.x, p3.y, p3.z);
            glEnd();
        }
    }
}


static void renderScene() {
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    setPerspective(70.0f, 0.05f, 200.0f);
    glLoadIdentity(); Vec3 center = camPos + forward(); lookAt(camPos, center, v3(0, 1, 0));

    glEnable(GL_TEXTURE_2D);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    glBindTexture(GL_TEXTURE_2D, texTile);
    drawTexturedQuad(v3(0, 0, 0), v3(COLS * CELL, 0, 0), v3(COLS * CELL, 0, ROWS * CELL), v3(0, 0, ROWS * CELL), 0, 0, COLS * 0.9f, ROWS * 0.9f);

    glBindTexture(GL_TEXTURE_2D, texCeil);
    drawTexturedQuad(v3(0, HGT, 0), v3(COLS * CELL, HGT, 0), v3(COLS * CELL, HGT, ROWS * CELL), v3(0, HGT, ROWS * CELL), 0, 0, COLS * 0.4f, ROWS * 0.4f);

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            const Cell& cell = grid[idx(c, r)]; float x = c * CELL, z = r * CELL;
            if (cell.wall[0]) drawWall(v3(x, 0, z), v3(x + CELL, 0, z));
            if (cell.wall[1]) drawWall(v3(x + CELL, 0, z), v3(x + CELL, 0, z + CELL));
            if (cell.wall[2]) drawWall(v3(x, 0, z + CELL), v3(x + CELL, 0, z + CELL));
            if (cell.wall[3]) drawWall(v3(x, 0, z), v3(x, 0, z + CELL));
        }
    }

    glDisable(GL_TEXTURE_2D);
}

static bool isFullscreen = false; static int winX = 0, winY = 0, winW = 1280, winH = 720; static int renderScale = 1;
static void toggleFullscreen(GLFWwindow* win) {
    isFullscreen = !isFullscreen;
    if (isFullscreen) { glfwGetWindowPos(win, &winX, &winY); glfwGetWindowSize(win, &winW, &winH); GLFWmonitor* m = glfwGetPrimaryMonitor(); const GLFWvidmode* md = glfwGetVideoMode(m); glfwSetWindowMonitor(win, m, 0, 0, md->width, md->height, md->refreshRate); W = md->width; H = md->height; }
    else { glfwSetWindowMonitor(win, nullptr, winX, winY, winW, winH, 0); W = winW; H = winH; }
}

static bool keys[512] = { 0 };
static void keycb(GLFWwindow* w, int key, int sc, int action, int mods) {
    if (key >= 0 && key < 512) { if (action == GLFW_PRESS) keys[key] = true; if (action == GLFW_RELEASE) keys[key] = false; }
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
        if (key == GLFW_KEY_R) generateMaze();
        if (key == GLFW_KEY_F) toggleFullscreen(w);
        if (key == GLFW_KEY_T) autoFly = !autoFly;
        if (key == GLFW_KEY_1) { texSize = std::max(16, texSize / 2); regenTextures(); }
        if (key == GLFW_KEY_2) { texSize = std::min(256, texSize * 2); regenTextures(); }
        if (key == GLFW_KEY_9) { renderScale = std::min(4, renderScale + 1); }
        if (key == GLFW_KEY_0) { renderScale = std::max(1, renderScale - 1); }

        if (key == GLFW_KEY_P) DISP_ON = !DISP_ON;         
        if (key == GLFW_KEY_LEFT_BRACKET) { DISP_SCALE = std::max(0.0f, DISP_SCALE - 0.01f); } 
        if (key == GLFW_KEY_RIGHT_BRACKET) { DISP_SCALE = std::min(0.25f, DISP_SCALE + 0.01f); } 
    }
}
static void mousebtn(GLFWwindow* w, int button, int action, int mods) { if (button == GLFW_MOUSE_BUTTON_LEFT) { dragging = (action == GLFW_PRESS) && !autoFly; if (dragging) glfwGetCursorPos(w, &mx, &my); } }
static void cursorpos(GLFWwindow* w, double x, double y) {
    if (dragging) {
        float dx = (float)(x - mx), dy = (float)(y - my);
        mx = x; my = y;
        yaw += dx * 0.005f;
        pitch += -dy * 0.005f;
        pitch = (pitch < -1.2f) ? -1.2f : (pitch > 1.2f ? 1.2f : pitch);
    }
}

int main() {
    s32 = (unsigned)std::time(nullptr);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    GLFWwindow* win = glfwCreateWindow(W, H, "3D Maze — Retro Bricks", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glfwSetKeyCallback(win, keycb); glfwSetMouseButtonCallback(win, mousebtn); glfwSetCursorPosCallback(win, cursorpos);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    generateMaze();
    regenTextures();

    camPos = v3(COLS * CELL * 0.5f, 0.4f, ROWS * CELL * 0.5f + 3.0f);
    yaw = 3.1415f; pitch = -0.04f;

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime(); float dt = (float)(now - last); last = now;

        if (autoFly) { updateAutoFly(dt); }
        else {
            float spd = 3.2f; Vec3 f = forward(); f.y = 0; f = norm(f); Vec3 r = rightv(); r.y = 0; r = norm(r);
            if (keys[GLFW_KEY_W]) camPos = camPos + f * (spd * dt);
            if (keys[GLFW_KEY_S]) camPos = camPos - f * (spd * dt);
            if (keys[GLFW_KEY_A]) camPos = camPos + r * (spd * dt);
            if (keys[GLFW_KEY_D]) camPos = camPos - r * (spd * dt);
        }

        glViewport(0, 0, W, H);
        glClearColor(0.85f, 0.75f, 0.25f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int vw = W / std::max(1, renderScale);
        int vh = H / std::max(1, renderScale);
        int vx = (W - vw) / 2;
        int vy = (H - vh) / 2;
        glViewport(vx, vy, vw, vh);
        renderScene();

        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, W, H, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnable(GL_SCISSOR_TEST);
        glScissor(vx, vy, vw, vh);

        float cx = vx + vw * 0.5f;
        float cy = vy + vh * 0.5f;

        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0, 0, 0, 0.0f); glVertex2f(cx, cy);
        glColor4f(0, 0, 0, 0.6f); glVertex2f((float)vx, (float)vy);
        glColor4f(0, 0, 0, 0.6f); glVertex2f((float)(vx + vw), (float)vy);
        glColor4f(0, 0, 0, 0.6f); glVertex2f((float)(vx + vw), (float)(vy + vh));
        glColor4f(0, 0, 0, 0.6f); glVertex2f((float)vx, (float)(vy + vh));
        glEnd();

        glDisable(GL_SCISSOR_TEST);

        glDisable(GL_BLEND);

        glViewport(0, 0, W, H);
        glfwSwapBuffers(win); glfwPollEvents();
    }

    glfwDestroyWindow(win); glfwTerminate(); return 0;
}
