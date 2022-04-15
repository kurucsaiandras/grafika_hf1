//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kurucsai Andras
// Neptun : WWEI3B
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

// 2D camera
class Camera2D {
	vec2 wCenter; // center in world coordinates
public:
	Camera2D() : wCenter(0, 0) { }

	void Pan(vec2 t) { wCenter = wCenter + t; }

	vec2 adjust(vec2 point) {
		return point - wCenter;
	}
};

Camera2D camera;		// 2D camera
GPUProgram gpuProgram; // vertex and fragment shaders

const int nv = 100; //kor es egyenes felbontasa
const float massH = 1.6735575E-27f; //hidrogen tomege //E-27f
const float chargeE = 1.60217662E-19f; //elektron toltese //E-19f
const float k = 8.988E9f; //Coulomb konstans
const float rho = 1E-20f; //MIGHT SCALE LATER

//implements Beltrami-Poincare projection
vec2 hyperproject(vec2 point) {
	float w = sqrtf(powf(point.x, 2.0f) + powf(point.y, 2.0f) + 1.0f);
	float x = point.x / (w + 1.0f);
	float y = point.y / (w + 1.0f);
	return vec2(x, y);
}

//rotates a point around a reference point
vec2 rotate(float phi, vec2 ref, vec2 point) {
	float x = (point - ref).x;
	float y = (point - ref).y;
	vec2 tmp = vec2(cosf(phi) * x - sinf(phi) * y, sinf(phi) * x + cosf(phi) * y);
	return (tmp + ref);
}



class Edge {
	unsigned int vao;
	vec2 p1, p2;
	vec2 euclidian_vertices[nv];
	vec2 refMassCentre; //center of mass of parent molecule
	float phi; // angle of rotation
	vec2 moleculePos;
public:
	Edge() {}
	void create(vec2 p1, vec2 p2) {
		phi = 0;
		moleculePos = vec2(0, 0);
		this->p1 = p1;
		this->p2 = p2;
		for (int i = 0; i < nv; i++) {
			vec2 t = i * (p2 - p1) / (nv - 1);
			euclidian_vertices[i] = p1 + t;
		}
	}

	void setMoleculePos(vec2 p) {
		this->moleculePos = p;
	}

	void setPhi(float p) {
		phi = p;
	}

	void setRef(vec2 r) {
		refMassCentre = r;
	}

	void Draw() {

		//itt rakjuk ra a vertices-re az összes transzformaciot
		//Forgatas -> Position -> poincare lekepzes
		vec2 vertices[nv];
		for (int i = 0; i < nv; i++) {
			vertices[i] = rotate(phi, refMassCentre, euclidian_vertices[i]);
			vertices[i] = vertices[i] + moleculePos;
			vertices[i] = camera.adjust(vertices[i]);
			vertices[i] = hyperproject(vertices[i]);
		}

		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			vertices,	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		//csucspont arnyalo 0. regiszterebe rakjuk az elso ket bajtot (vec4 struct. x és y -jat tolti ki)
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, nv /*# Elements*/);
	}
};

class Atom {
	unsigned int vao;	   // virtual world on the GPU
	float r;	  //radius of circle
	vec2 position; //position of the center of the circle
	vec2 euclidian_vertices[nv];
	float mass;
	float charge;
	vec2 refMassCentre; //center of mass of parent molecule
	float phi; // angle of rotation
	vec2 moleculePos;
public:
	Atom() {};
	void create(vec2 pos) {
		r = 0.05f;
		position = pos;
		moleculePos = vec2(0, 0);
		mass = massH * static_cast <float> (rand());
		charge = chargeE * (static_cast <float> (rand()) * 2.0f - static_cast <float> (RAND_MAX));
		printf("charge: %f\n", charge*10e19);

		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			float x = r * cosf(fi);
			float y = r * sinf(fi);
			euclidian_vertices[i] = vec2(x, y);
		}

	}

	void setPhi(float p) {
		this->phi = p;
	}

	float getPhi() {
		return this->phi;
	}

	void setRef(vec2 r) {
		refMassCentre = r;
	}

	vec2 getPos() {
		return position;
	}

	float getCharge() {
		return charge;
	}

	void setCharge(float c) {
		charge = c;
	}

	float getMass() {
		return mass;
	}

	void setMoleculePos(vec2 p) {
		this->moleculePos = p;
	}

	//mat4 M() {
	//	int sx = 1, sy = 1, sw = 1, phi = 0; //TODO: ezeket animacioban allitani
	//	mat4 Mscale(sx, 0, 0, 0,
	//		0, sy, 0, 0,
	//		0, 0, sw, 0,
	//		0, 0, 0, 1); // scaling

	//	mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
	//		-sinf(phi), cosf(phi), 0, 0,
	//		0, 0, 1, 0,
	//		0, 0, 0, 1); // rotation

	//	mat4 Mtranslate(1, 0, 0, 0,
	//		0, 1, 0, 0,
	//		0, 0, 1, 0,
	//		position.x, position.y, 0, 1); // translation

	//	return /*Mscale * Mrotate * */Mtranslate;	// model transformation
	//}

	//applies position
	vec2 getvertexposition(vec2 point) {
		return vec2(point.x + position.x, point.y + position.y);
	}

	void Draw() {

		//itt rakjuk ra a vertices-re az összes transzformaciot
		//Forgatas -> Position -> poincare lekepzes
		vec2 vertices[nv];
		for (int i = 0; i < nv; i++) {
			vertices[i] = getvertexposition(euclidian_vertices[i]);
			vertices[i] = rotate(phi, refMassCentre, vertices[i]);
			vertices[i] = vertices[i] + moleculePos;
			vertices[i] = camera.adjust(vertices[i]);
			vertices[i] = hyperproject(vertices[i]);
		}

		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			vertices,	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		//csucspont arnyalo 0. regiszterebe rakjuk az elso ket bajtot (vec4 struct. x és y -jat tolti ki)
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		float R, B;
		if (charge < 0) {
			R = 0.0f;
			B = -1.0 * charge / chargeE / RAND_MAX;
		}
		else {
			R = charge / chargeE / RAND_MAX;
			B = 0.0f;
		}
		glUniform3f(location, R, 0.0f, B); // 3 floats RED

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nv /*# Elements*/);
	}
};

class Molecule {
	unsigned int atomsNum;
	std::vector<Atom> atoms;
	std::vector<Edge> edges;
	vec2 massCentre;
	vec2 v; //velocity
	vec3 w; //angular velocity
	vec2 pos;
	float phi;
	float tprev;
public:
	Molecule() {};

	vec2 generateNeighbourPos(vec2 pos)
	{
		float angle = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * 2.0f * M_PI; //random angle between 0 and 2pi
		float dist = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * 0.15f + 0.2f; //random distance between 0.2 and 0.35
		printf("edge length: %f\n", dist);
		return vec2(pos.x + dist * cosf(angle), pos.y + dist * sinf(angle));
	}

	void create() {
		atoms.clear();
		edges.clear();
		pos = vec2(0, 0);
		phi = 0;
		w = vec3(0, 0, 0);
		v = vec2(0, 0);
		tprev = 0;
		atomsNum = rand() % 7 + 2; //random num between 2 and 8
		Atom firstatom;
		float x = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2)); //camerapos-hoz adjustolni esetleg??
		float y = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));
		firstatom.create(vec2(x, y));
		atoms.insert(atoms.end(), firstatom);
		int needMore = atomsNum-1;
		int currentAtomIdx = 0;
		while (needMore != 0)
		{
			int neighboursNum = rand() % 3 + 1; //random number between 1 and 3
			if (neighboursNum > needMore)
				neighboursNum = needMore;
			vec2 pos = atoms[currentAtomIdx].getPos();
			for (int j = 0; j < neighboursNum; j++)
			{
				vec2 npos = generateNeighbourPos(pos);

				Atom tmpatom;
				tmpatom.create(npos);
				atoms.insert(atoms.end(), tmpatom);

				Edge tmpedge;
				tmpedge.create(pos, npos);
				edges.insert(edges.end(), tmpedge);
			}
			needMore -= neighboursNum;
			currentAtomIdx++;
		}

		//adjust the last atoms charge (sum of charge must be zero)
		float chargeSum = 0;
		for (auto it = begin(atoms); it != end(atoms) - 1; it++)
		{
			chargeSum += it->getCharge();
		}
		atoms[atoms.size()-1].setCharge(chargeSum * (-1.0f));


		//calculate centre of mass
		vec2 Snum = vec2(0, 0);
		float Sden = (0, 0);
		for (Atom a : atoms)
		{
			Snum = Snum + a.getMass() * a.getPos();
			Sden = Sden + a.getMass();
		}
		massCentre = Snum / Sden;
		for (auto it = begin(edges); it != end(edges); it++) {
			it->setRef(massCentre);
		}
		for (auto it = begin(atoms); it != end(atoms); it++) {
			it->setRef(massCentre);
		}
	}
	
	void Animate(float t, Molecule other)
	{
		//float dt = t - tprev;
		float dt = 0.01f;
		tprev = t;
		printf("DTime: %f\n", dt);
		vec2 sumF = vec2(0, 0);
		vec3 sumM = vec3(0, 0, 0);
		float sumTheta = 0;
		float sumMass = 0;
		vec2 m1massPos = massCentre + pos;
		for (auto i = begin(atoms); i != end(atoms); i++) {
			vec2 posi = rotate(i->getPhi(), massCentre, i->getPos()) + pos; //get absolute position

			vec3 r3D = vec3(posi.x - m1massPos.x, posi.y - m1massPos.y, 0); //helyvektor a sulyponthoz kepest
			vec3 v3D = cross(w, r3D);
			vec2 vi = vec2(v.x + v3D.x, v.y + v3D.y); //i atom sebessege
			printf("i atom velocity: %f %f\n", vi.x, vi.y);
			vec2 i_kozegF = -1.0f * rho * vi; //kozegellenallas
			printf("Kozegell: %f %f\n", i_kozegF.x, i_kozegF.y);
			vec2 i_coulombForce = vec2(0, 0);
			for (auto j = begin(other.atoms); j != end(other.atoms); j++) {
				vec2 posj = rotate(j->getPhi(), other.massCentre, j->getPos()) + other.pos; //get absolute position

				vec2 e = normalize(posi - posj); //unit vector
				float d = length(posj - posi);
				//avoid singularity ????
				if (d < 0.1) {
					d = 0.1;
				}
				vec2 cF = (k * i->getCharge() * j->getCharge() / d) * e; //coulomb force formula

				i_coulombForce = i_coulombForce + cF;
			}
			printf("Coulomb: %f %f\n", i_coulombForce.x, i_coulombForce.y);
			vec2 i_F = i_kozegF + i_coulombForce; //i atomra hato ossz ero
			vec3 i_M = cross(r3D, vec3(i_F.x, i_F.y, 0)); //i atomra hato nyomatek
			float i_theta = i->getMass() * dot(r3D, r3D);
			sumF = sumF + i_F;
			sumM = sumM + i_M;
			sumTheta += i_theta;
			sumMass += i->getMass();
		}
		vec2 dv = (dt / sumMass) * sumF * 0.001;
		v = v + dv;
		printf("Velocity Diff: %f %f\n", dv.x, dv.y);
		vec3 dw = (dt / sumTheta) * sumM * 0.001;
		w = w + dw;

		pos = pos + v * dt;
		printf("DPOS: %f %f\n", (v * dt).x, (v * dt).y); //azonnal elszáll a végtelenbe :(
		phi = phi + w.z * dt;
		for (auto it = begin(edges); it != end(edges); it++) {
			it->setPhi(phi);
			it->setMoleculePos(pos);
		}
		for (auto it = begin(atoms); it != end(atoms); it++) {
			it->setPhi(phi);
			it->setMoleculePos(pos);
		}
	}

	void Draw()
	{
		for (Edge e : edges) e.Draw();
		for (Atom a : atoms) a.Draw();
	}
};

Molecule molecule1, molecule2;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	molecule1.create();
	molecule2.create();
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");

	mat4 MVPTransform = mat4(1, 0, 0, 0,    // MVP matrix
		0, 1, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 1);
	gpuProgram.setUniform(MVPTransform, "MVP");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.3, 0.3, 0.3, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	molecule1.Draw();
	molecule2.Draw();

	glutSwapBuffers();// exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case ' ': molecule1.create(); molecule2.create(); break;
	case 's': camera.Pan(vec2(-0.1, 0)); break;
	case 'd': camera.Pan(vec2(+0.1, 0)); break;
	case 'e': camera.Pan(vec2(0, 0.1)); break;
	case 'x': camera.Pan(vec2(0, -0.1)); break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	molecule1.Animate(sec, molecule2);					// animate the triangle object
	molecule2.Animate(sec, molecule1);
	glutPostRedisplay();
}
