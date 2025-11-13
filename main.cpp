#include <GLFW/glfw3.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <string>

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

enum GameState { MENU, PLAYING };
static GameState gameState = MENU;

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

static Vec3 camPos;
static float yaw = 0.0f, pitch = 0.0f;
static double lastMouseX = 0, lastMouseY = 0;
static bool firstMouse = true;

static Vec3 forward() { return v3(std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch)); }
static Vec3 rightv() { Vec3 f = forward(); return v3(f.z, 0, -f.x); }

static bool checkCollision(Vec3 pos) {
    float margin = 0.3f;
    int c = (int)(pos.x / CELL);
    int r = (int)(pos.z / CELL);

    if (c < 0 || c >= COLS || r < 0 || r >= ROWS) return true;

    float lx = c * CELL;
    float rx = (c + 1) * CELL;
    float tz = r * CELL;
    float bz = (r + 1) * CELL;

    const Cell& cell = grid[idx(c, r)];
    if (cell.wall[0] && pos.z < tz + margin) return true; 
    if (cell.wall[1] && pos.x > rx - margin) return true; 
    if (cell.wall[2] && pos.z > bz - margin) return true; 
    if (cell.wall[3] && pos.x < lx + margin) return true; 

    return false;
}

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

static GLuint texBrick = 0, texTile = 0, texCeil = 0;
static int texSize = 64;

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

static inline float brickHeight(float u, float v) {
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

static void drawWall(Vec3 a, Vec3 b) {
    Vec3 t = b - a;
    Vec3 up = v3(0, 1, 0);
    Vec3 n = norm(cross(up, t));
    glBindTexture(GL_TEXTURE_2D, texBrick);
    float lenW = len(t);

    const int SX = 8;  
    const int SY = 8;  

    const float uMax = lenW * 0.8f;
    const float vMax = HGT * 0.8f;

    for (int iy = 0; iy < SY; ++iy) {
        for (int ix = 0; ix < SX; ++ix) {
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

            if (DISP_ON && DISP_SCALE > 0.0f) {
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

static bool isWallVisible(Vec3 a, Vec3 b, Vec3 camPos, Vec3 camForward) {
    Vec3 wallCenter = (a + b) * 0.5f;
    wallCenter.y = HGT * 0.5f;
    Vec3 toWall = wallCenter - camPos;
    return dot(norm(toWall), camForward) > -0.3f; 
}

static void renderScene() {
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    setPerspective(70.0f, 0.05f, 200.0f);
    glLoadIdentity();
    Vec3 center = camPos + forward();
    lookAt(camPos, center, v3(0, 1, 0));

    glEnable(GL_TEXTURE_2D);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    Vec3 camForward = forward();

    glBindTexture(GL_TEXTURE_2D, texTile);
    drawTexturedQuad(v3(0, 0, 0), v3(COLS * CELL, 0, 0), v3(COLS * CELL, 0, ROWS * CELL), v3(0, 0, ROWS * CELL), 0, 0, COLS * 0.9f, ROWS * 0.9f);

    glBindTexture(GL_TEXTURE_2D, texCeil);
    drawTexturedQuad(v3(0, HGT, 0), v3(COLS * CELL, HGT, 0), v3(COLS * CELL, HGT, ROWS * CELL), v3(0, HGT, ROWS * CELL), 0, 0, COLS * 0.4f, ROWS * 0.4f);

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            const Cell& cell = grid[idx(c, r)];
            float x = c * CELL, z = r * CELL;

            if (cell.wall[0]) {
                Vec3 a = v3(x, 0, z), b = v3(x + CELL, 0, z);
                if (isWallVisible(a, b, camPos, camForward)) drawWall(a, b);
            }
            if (cell.wall[1]) {
                Vec3 a = v3(x + CELL, 0, z), b = v3(x + CELL, 0, z + CELL);
                if (isWallVisible(a, b, camPos, camForward)) drawWall(a, b);
            }
            if (cell.wall[2]) {
                Vec3 a = v3(x, 0, z + CELL), b = v3(x + CELL, 0, z + CELL);
                if (isWallVisible(a, b, camPos, camForward)) drawWall(a, b);
            }
            if (cell.wall[3]) {
                Vec3 a = v3(x, 0, z), b = v3(x, 0, z + CELL);
                if (isWallVisible(a, b, camPos, camForward)) drawWall(a, b);
            }
        }
    }

    glDisable(GL_TEXTURE_2D);
}

static void drawMinimap() {
    float mapSize = 200.0f;
    float mapX = W - mapSize - 20.0f;
    float mapY = 20.0f;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.1f, 0.1f, 0.1f, 0.8f);
    glBegin(GL_QUADS);
    glVertex2f(mapX, mapY);
    glVertex2f(mapX + mapSize, mapY);
    glVertex2f(mapX + mapSize, mapY + mapSize);
    glVertex2f(mapX, mapY + mapSize);
    glEnd();

    glColor4f(0.9f, 0.9f, 0.9f, 1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(mapX, mapY);
    glVertex2f(mapX + mapSize, mapY);
    glVertex2f(mapX + mapSize, mapY + mapSize);
    glVertex2f(mapX, mapY + mapSize);
    glEnd();

    float cellW = mapSize / COLS;
    float cellH = mapSize / ROWS;

    glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
    glBegin(GL_LINES);
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            const Cell& cell = grid[idx(c, r)];
            float x = mapX + c * cellW;
            float y = mapY + r * cellH;

            if (cell.wall[0]) { 
                glVertex2f(x, y);
                glVertex2f(x + cellW, y);
            }
            if (cell.wall[1]) { 
                glVertex2f(x + cellW, y);
                glVertex2f(x + cellW, y + cellH);
            }
            if (cell.wall[2]) { 
                glVertex2f(x, y + cellH);
                glVertex2f(x + cellW, y + cellH);
            }
            if (cell.wall[3]) { 
                glVertex2f(x, y);
                glVertex2f(x, y + cellH);
            }
        }
    }
    glEnd();

    float px = mapX + (camPos.x / (COLS * CELL)) * mapSize;
    float py = mapY + (camPos.z / (ROWS * CELL)) * mapSize;

    glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
    glPointSize(8.0f);
    glBegin(GL_POINTS);
    glVertex2f(px, py);
    glEnd();

    float dirLen = 15.0f;
    float dx = std::sin(yaw) * dirLen;
    float dy = std::cos(yaw) * dirLen;
    glBegin(GL_LINES);
    glVertex2f(px, py);
    glVertex2f(px + dx, py + dy);
    glEnd();

    glDisable(GL_BLEND);
}

static const unsigned char font5x7[96][7] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space
    {0x04,0x04,0x04,0x04,0x00,0x04,0x00}, // !
    {0x0A,0x0A,0x00,0x00,0x00,0x00,0x00}, // "
    {0x0A,0x1F,0x0A,0x1F,0x0A,0x00,0x00}, // #
    {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04}, // $
    {0x18,0x19,0x02,0x04,0x08,0x13,0x03}, // %
    {0x0C,0x12,0x14,0x08,0x15,0x12,0x0D}, // &
    {0x04,0x04,0x00,0x00,0x00,0x00,0x00}, // '
    {0x02,0x04,0x08,0x08,0x08,0x04,0x02}, // (
    {0x08,0x04,0x02,0x02,0x02,0x04,0x08}, // )
    {0x00,0x0A,0x04,0x1F,0x04,0x0A,0x00}, // *
    {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, // +
    {0x00,0x00,0x00,0x00,0x04,0x04,0x08}, // ,
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, // -
    {0x00,0x00,0x00,0x00,0x00,0x04,0x00}, // .
    {0x00,0x01,0x02,0x04,0x08,0x10,0x00}, // /
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // 2
    {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}, // 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
    {0x00,0x04,0x00,0x00,0x00,0x04,0x00}, // :
    {0x00,0x04,0x00,0x00,0x04,0x04,0x08}, // ;
    {0x02,0x04,0x08,0x10,0x08,0x04,0x02}, // <
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // =
    {0x08,0x04,0x02,0x01,0x02,0x04,0x08}, // >
    {0x0E,0x11,0x01,0x02,0x04,0x00,0x04}, // ?
    {0x0E,0x11,0x17,0x15,0x17,0x10,0x0E}, // @
    {0x0E,0x11,0x11,0x11,0x1F,0x11,0x11}, // A
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C
    {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}, // D
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // E
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // F
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, // G
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, // H
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, // I
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, // J
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // K
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, // L
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, // M
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // N
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // O
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, // P
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, // Q
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // R
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, // S
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // T
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, // U
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, // V
    {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}, // W
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, // X
    {0x11,0x11,0x11,0x0A,0x04,0x04,0x04}, // Y
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // Z
    {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E}, // [
    {0x00,0x10,0x08,0x04,0x02,0x01,0x00}, // backslash
    {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E}, // ]
    {0x04,0x0A,0x11,0x00,0x00,0x00,0x00}, // ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x1F}, // _
};

static void drawChar(float x, float y, char c, float scale, float r, float g, float b) {
    if (c < 32 || c > 127) return;
    const unsigned char* glyph = font5x7[c - 32];

    glColor3f(r, g, b);
    glBegin(GL_QUADS);

    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (glyph[row] & (1 << (4 - col))) {
                float px = x + col * scale;
                float py = y + row * scale;
                glVertex2f(px, py);
                glVertex2f(px + scale, py);
                glVertex2f(px + scale, py + scale);
                glVertex2f(px, py + scale);
            }
        }
    }

    glEnd();
}

static void drawText(float x, float y, const char* text, float scale, float r, float g, float b) {
    float offset = 0;
    for (const char* c = text; *c != '\0'; c++) {
        drawChar(x + offset, y, *c, scale, r, g, b);
        offset += 6 * scale;
    }
}

static void drawMenu() {
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, W, H, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);
    glColor3f(0.1f, 0.1f, 0.15f);
    glVertex2f(0, 0);
    glVertex2f((float)W, 0);
    glColor3f(0.2f, 0.15f, 0.25f);
    glVertex2f((float)W, (float)H);
    glVertex2f(0, (float)H);
    glEnd();

    float titleY = H * 0.15f;
    const char* title = "3D MAZE";
    float titleScale = 8.0f;
    float titleWidth = 42 * titleScale; 
    float titleX = W * 0.5f - titleWidth * 0.5f;

    drawText(titleX + 4, titleY + 4, title, titleScale, 0.0f, 0.0f, 0.0f);
    drawText(titleX, titleY, title, titleScale, 0.95f, 0.75f, 0.3f);

    const char* subtitle = "RETRO BRICKS";
    float subScale = 3.0f;
    float subWidth = 72 * subScale;
    float subX = W * 0.5f - subWidth * 0.5f;
    drawText(subX, titleY + 80, subtitle, subScale, 0.7f, 0.7f, 0.7f);

    float btnW = 320.0f;
    float btnH = 70.0f;
    float btnX = W * 0.5f - btnW * 0.5f;
    float btnY = H * 0.45f;

    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_QUADS);
    glVertex2f(btnX + 5, btnY + 5);
    glVertex2f(btnX + btnW + 5, btnY + 5);
    glVertex2f(btnX + btnW + 5, btnY + btnH + 5);
    glVertex2f(btnX + 5, btnY + btnH + 5);
    glEnd();

    glColor3f(0.25f, 0.55f, 0.25f);
    glBegin(GL_QUADS);
    glVertex2f(btnX, btnY);
    glVertex2f(btnX + btnW, btnY);
    glVertex2f(btnX + btnW, btnY + btnH);
    glVertex2f(btnX, btnY + btnH);
    glEnd();

    glColor3f(0.9f, 0.95f, 0.9f);
    glLineWidth(4.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(btnX, btnY);
    glVertex2f(btnX + btnW, btnY);
    glVertex2f(btnX + btnW, btnY + btnH);
    glVertex2f(btnX, btnY + btnH);
    glEnd();

    const char* startText = "PRESS SPACE TO START";
    float startScale = 3.0f;
    float startWidth = 120 * startScale;
    float startX = W * 0.5f - startWidth * 0.5f;
    drawText(startX, btnY + 22, startText, startScale, 1.0f, 1.0f, 1.0f);

    float instrY = H * 0.60f;
    float lineHeight = 28.0f;
    float instrScale = 2.2f;

    drawText(W * 0.5f - 150, instrY, "CONTROLS:", instrScale, 0.95f, 0.75f, 0.3f);
    instrY += lineHeight + 10;

    const char* instructions[] = {
        "WASD - Move",
        "MOUSE - Look around",
        "R - Regenerate maze",
        "P - Toggle displacement",
        "[ ] - Displacement scale",
        "1/2 - Texture size",
        "9/0 - Render scale",
        "F - Fullscreen",
        "ESC - Menu/Exit"
    };

    for (int i = 0; i < 9; i++) {
        drawText(W * 0.5f - 200, instrY + i * lineHeight, instructions[i], instrScale, 0.85f, 0.85f, 0.85f);
    }
}

static bool isFullscreen = false;
static int winX = 0, winY = 0, winW = 1280, winH = 720;
static int renderScale = 1;

static void toggleFullscreen(GLFWwindow* win) {
    isFullscreen = !isFullscreen;
    if (isFullscreen) {
        glfwGetWindowPos(win, &winX, &winY);
        glfwGetWindowSize(win, &winW, &winH);
        GLFWmonitor* m = glfwGetPrimaryMonitor();
        const GLFWvidmode* md = glfwGetVideoMode(m);
        glfwSetWindowMonitor(win, m, 0, 0, md->width, md->height, md->refreshRate);
        W = md->width; H = md->height;
    }
    else {
        glfwSetWindowMonitor(win, nullptr, winX, winY, winW, winH, 0);
        W = winW; H = winH;
    }
}

static bool keys[512] = { 0 };
static bool menuClickHandled = false;

static void keycb(GLFWwindow* w, int key, int sc, int action, int mods) {
    if (key >= 0 && key < 512) {
        if (action == GLFW_PRESS) keys[key] = true;
        if (action == GLFW_RELEASE) keys[key] = false;
    }

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            if (gameState == PLAYING) {
                gameState = MENU;
                glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            else {
                glfwSetWindowShouldClose(w, 1);
            }
        }

        if (gameState == MENU) {
            if (key == GLFW_KEY_SPACE || key == GLFW_KEY_ENTER) {
                gameState = PLAYING;
                glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                firstMouse = true;
            }
        }

        if (gameState == PLAYING) {
            if (key == GLFW_KEY_R) generateMaze();
            if (key == GLFW_KEY_F) toggleFullscreen(w);
            if (key == GLFW_KEY_1) { texSize = std::max(16, texSize / 2); regenTextures(); }
            if (key == GLFW_KEY_2) { texSize = std::min(256, texSize * 2); regenTextures(); }
            if (key == GLFW_KEY_9) { renderScale = std::min(4, renderScale + 1); }
            if (key == GLFW_KEY_0) { renderScale = std::max(1, renderScale - 1); }
            if (key == GLFW_KEY_P) DISP_ON = !DISP_ON;
            if (key == GLFW_KEY_LEFT_BRACKET) { DISP_SCALE = std::max(0.0f, DISP_SCALE - 0.01f); }
            if (key == GLFW_KEY_RIGHT_BRACKET) { DISP_SCALE = std::min(0.25f, DISP_SCALE + 0.01f); }
        }
    }
}

static void mousebtn(GLFWwindow* w, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        if (gameState == MENU && !menuClickHandled) {
            gameState = PLAYING;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            firstMouse = true;
            menuClickHandled = true;
        }
    }
    if (action == GLFW_RELEASE) {
        menuClickHandled = false;
    }
}

static void cursorpos(GLFWwindow* w, double x, double y) {
    if (gameState != PLAYING) return;

    if (firstMouse) {
        lastMouseX = x;
        lastMouseY = y;
        firstMouse = false;
    }

    float dx = (float)(x - lastMouseX);
    float dy = (float)(y - lastMouseY);
    lastMouseX = x;
    lastMouseY = y;

    float sensitivity = 0.002f;
    yaw += dx * sensitivity;
    pitch -= dy * sensitivity;

    if (pitch > 1.5f) pitch = 1.5f;
    if (pitch < -1.5f) pitch = -1.5f;
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

    GLFWwindow* win = glfwCreateWindow(W, H, "3D Maze - Retro Bricks", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glfwSetKeyCallback(win, keycb);
    glfwSetMouseButtonCallback(win, mousebtn);
    glfwSetCursorPosCallback(win, cursorpos);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    generateMaze();
    regenTextures();

    camPos = v3(COLS * CELL * 0.5f, 0.4f, ROWS * CELL * 0.5f);
    yaw = 3.1415f;
    pitch = 0.0f;

    double last = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        if (gameState == PLAYING) {
            float spd = 3.2f;
            Vec3 f = forward();
            f.y = 0;
            f = norm(f);
            Vec3 r = rightv();
            r.y = 0;
            r = norm(r);

            Vec3 newPos = camPos;
            if (keys[GLFW_KEY_W]) newPos = newPos + f * (spd * dt);
            if (keys[GLFW_KEY_S]) newPos = newPos - f * (spd * dt);
            if (keys[GLFW_KEY_A]) newPos = newPos - r * (spd * dt);
            if (keys[GLFW_KEY_D]) newPos = newPos + r * (spd * dt);

            if (!checkCollision(newPos)) {
                camPos = newPos;
            }
        }

        glViewport(0, 0, W, H);
        glClearColor(0.85f, 0.75f, 0.25f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (gameState == MENU) {
            drawMenu();
        }
        else {
            int vw = W / std::max(1, renderScale);
            int vh = H / std::max(1, renderScale);
            int vx = (W - vw) / 2;
            int vy = (H - vh) / 2;
            glViewport(vx, vy, vw, vh);
            renderScene();

            glDisable(GL_DEPTH_TEST);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, W, H, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            float vignetteSize = std::min(vw, vh) * 0.3f;

            glBegin(GL_QUADS);
            glColor4f(0, 0, 0, 0.5f);
            glVertex2f((float)vx, (float)vy);
            glVertex2f((float)(vx + vw), (float)vy);
            glColor4f(0, 0, 0, 0.0f);
            glVertex2f((float)(vx + vw), (float)(vy + vignetteSize));
            glVertex2f((float)vx, (float)(vy + vignetteSize));
            glEnd();

            glBegin(GL_QUADS);
            glColor4f(0, 0, 0, 0.0f);
            glVertex2f((float)vx, (float)(vy + vh - vignetteSize));
            glVertex2f((float)(vx + vw), (float)(vy + vh - vignetteSize));
            glColor4f(0, 0, 0, 0.5f);
            glVertex2f((float)(vx + vw), (float)(vy + vh));
            glVertex2f((float)vx, (float)(vy + vh));
            glEnd();

            glBegin(GL_QUADS);
            glColor4f(0, 0, 0, 0.5f);
            glVertex2f((float)vx, (float)vy);
            glColor4f(0, 0, 0, 0.0f);
            glVertex2f((float)(vx + vignetteSize), (float)vy);
            glVertex2f((float)(vx + vignetteSize), (float)(vy + vh));
            glColor4f(0, 0, 0, 0.5f);
            glVertex2f((float)vx, (float)(vy + vh));
            glEnd();

            glBegin(GL_QUADS);
            glColor4f(0, 0, 0, 0.0f);
            glVertex2f((float)(vx + vw - vignetteSize), (float)vy);
            glColor4f(0, 0, 0, 0.5f);
            glVertex2f((float)(vx + vw), (float)vy);
            glVertex2f((float)(vx + vw), (float)(vy + vh));
            glColor4f(0, 0, 0, 0.0f);
            glVertex2f((float)(vx + vw - vignetteSize), (float)(vy + vh));
            glEnd();

            glDisable(GL_BLEND);

            glViewport(0, 0, W, H);
            drawMinimap();
        }

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}