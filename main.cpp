#include <stdcpp.h>
#include "/opt/homebrew/Cellar/open-mpi/4.1.5/include/mpi.h"

using namespace std;
using namespace std::chrono;

#pragma region Matrices

double** A;
double** B;
double** C;
double** D;
double** E;
double** G;
double** T0;
double** T1;
double** T2;

#pragma endregion

#pragma region Functions
// Function to measure the execution time of a function
template<typename Func>
double measureExecutionTime(Func func) {
    auto startTime = high_resolution_clock::now();
    func();
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime).count();
    return duration / 1e6; // Convert to milliseconds
}

double** createMatrix(int size) {
    srand(time(0)); // Seed the random number generator with the current time
    double** matrix = new double*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (rand() % 100) / 100.0; // Generate random numbers between 0 and 9.9
//            matrix[i][j] = i + j + 1.5;
        }
    }
    return matrix;
}

void deleteMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void addMatrices(double** matrix1, double** matrix2, double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void subtractMatrices(double** matrix1, double** matrix2, double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}

void multiplyMatrices(double** matrix1, double** matrix2, double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = matrix1[i][j] * matrix2[i][j];
//            double sum = 0;
//            for (int k = 0; k < size; k += 4) {
//                sum += matrix1[i][k] * matrix2[k][j];
//                sum += matrix1[i][k + 1] * matrix2[k + 1][j];
//                sum += matrix1[i][k + 2] * matrix2[k + 2][j];
//                sum += matrix1[i][k + 3] * matrix2[k + 3][j];
//            }
//            result[i][j] = sum;
        }
    }
}

void divideMatrices(double** matrix1, double** matrix2, double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = matrix1[i][j] / matrix2[i][j];
        }
    }
}

void printMatrices(double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << result[i][j] << " ";
        }
        cout << endl;
    }
}

void init (const int & size)
{
    A = createMatrix(size);
    B = createMatrix(size);
    C = createMatrix(size);
    D = createMatrix(size);
    E = createMatrix(size);
    G = createMatrix(size);
    T0 = createMatrix(size);
    T1 = createMatrix(size);
    T2 = createMatrix(size);
}

void finalize (const int & size)
{
    deleteMatrix(A, size);
    deleteMatrix(B, size);
    deleteMatrix(C, size);
    deleteMatrix(D, size);
    deleteMatrix(E, size);
    deleteMatrix(G, size);
    deleteMatrix(T0, size);
    deleteMatrix(T1, size);
    deleteMatrix(T2, size);
}

#pragma endregion

void run (const int & size, int wRank, int wSize)
{
    ofstream outputFile("mpiResults.txt", std::ios::app);
    init(size);
    double mpiTime0, mpiTime1, mpiTime2;
    MPI_Status status;
    cout << "DEBUG MODE" << endl;


    switch (wRank) {
        case 0:
        {
            cout << "rank 0 begins" << endl;
            subtractMatrices(A, E, T0, size);   // T0 = A - E
            cout << " R0: A - E done" << endl;
            divideMatrices(T0, G, T1, size);    // T1 = (A - E) / G
            cout << " R0: :T0 / G done" << endl;
            addMatrices(T1, C, T2, size);       // T2 = (A - E) / G + C
            cout << " R0: T1 + C done" << endl;
            multiplyMatrices(T2, T2, T0, size); // Y1 = ((A - E) / G + C)^2
            cout << " R0: T2 * T2 done" << endl;
            MPI_Send(&(T0[0][0]), size*size, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);   // Y1 -> 2ому процессу
            cout << " R0: Y1 -> 2 done" << endl;
            MPI_Send(&(T0[0][0]), size*size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);   // Y1 -> 1ому процессу
            cout << " R0: Y1 -> 1 done" << endl;
            cout << "rank 0 ends" << endl;
            break;
        }

        case 1:
        {
            cout << "rank 1 begins" << endl;
            mpiTime1 = MPI_Wtime();
            multiplyMatrices(B, D, T0, size);       // T0 = B * D
            cout << " R1: B * D done" << endl;
            subtractMatrices(T0, E, T1, size);      // T1 = BD - E
            cout << " R1: T0 - E done" << endl;
            multiplyMatrices(T1, T1, T2, size);     // T2 = (BD - E)^2
            cout << " R1: T1 * T1 done" << endl;
            MPI_Recv(&(A[0][0]), size*size, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &status);   // 2ой процесс -> T1 = (A - E) * G
            cout << " R1: 2 -> A done" << endl;
            addMatrices(A, C, T0, size);            // T0 = (A - E) * G + C
            cout << " R1: A + C done" << endl;
            multiplyMatrices(T0, T2, T1, size);     // T4 = ((A - E) * G + C) * (BD - E)^2
            cout << " R1: T0 * T2 done" << endl;
            MPI_Recv(&(G[0][0]), size*size, MPI_DOUBLE, 3, 5, MPI_COMM_WORLD, &status);   // 3ий процесс -> T4 = (A - E) * G * ((BD - E) - G) * BD
            cout << " R1: 3 -> G done" << endl;
            multiplyMatrices(G, T1, T2, size);  // Y3 = (A - E) * G * ((BD - E) - G) * BD * ((A - E) * G + C) * (BD - E)^2
            cout << " R1: G * T1 done" << endl;
            MPI_Recv(&(T0[0][0]), size*size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status); // Y1 = ((A - E) / G + C)^2
            cout << " R1: 0 -> T0 done" << endl;
            MPI_Recv(&(T1[0][0]), size*size, MPI_DOUBLE, 2, 4, MPI_COMM_WORLD, &status); // Y2 = Y1 - (A - E) * G + C
            cout << " R1: 2 -> T1 done" << endl;
            mpiTime2 = MPI_Wtime();
            mpiTime0 = mpiTime2 - mpiTime1;
            if (outputFile.is_open())
            {
                outputFile << "Matrix size is: " << size << endl << "Execution time is: " << mpiTime0 << " seconds" << endl << "----------------------------------------------" << endl;
            }
            cout << "rank 1 ends" << endl;
            break;
        }
        case 2:
        {
            cout << "rank 2 begins" << endl;
            subtractMatrices(A, E, T0, size);       // T0 = A - E
            cout << " R2: A / E done" << endl;
            multiplyMatrices(T0, G, T1, size);      // T1 = (A - E) * G
            cout << " R2: T0 * G done" << endl;
            MPI_Send(&(T1[0][0]), size*size, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);   // T1 = (A - E) * G -> 1ому процессу
            cout << " R2: T1 -> 1 done" << endl;
            MPI_Send(&(T1[0][0]), size*size, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD);   // T1 = (A - E) * G -> 3ому процессу
            cout << " R2: T1 -> 3 done" << endl;
            subtractMatrices(C, T1, T2, size);      // T2 = C - (A - E) * G
            cout << " R2: C - T1 done" << endl;
            MPI_Recv(&(B[0][0]), size*size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status); // Y1 = ((A - E) / G + C)^2
            cout << " R2: 0 -> B done" << endl;
            addMatrices(T2, B, T0, size);           // Y2 = Y1 - (A - E) * G + C
            cout << " R2: T2 + B done" << endl;
            MPI_Send(&(T0[0][0]), size*size, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD); // Y2 = Y1 - (A - E) * G + C -> 1ому процессу
            cout << " R2: T0 -> 1 done" << endl;
            cout << "rank 2 ends" << endl;
            break;
        }
        case 3:
        {
            cout << "rank 3 begins" << endl;
            multiplyMatrices(B, D, T0, size);       // T0 = B * D
            cout << " R3: B * D done" << endl;
            subtractMatrices(T0, E, T1, size);      // T1 = BD - E
            cout << " R3: T0 - E done" << endl;
            subtractMatrices(T1, G, T2, size);      // T2 = (BD - E) - G
            cout << " R3: T1 - G done" << endl;
            MPI_Recv(&(A[0][0]), size*size, MPI_DOUBLE, 2, 3, MPI_COMM_WORLD, &status);   // 2ой процесс -> T1 = (A - E) * G
            cout << " R3: 2 -> A done" << endl;
            multiplyMatrices(A, T2, T1, size);      // T3 = (A - E) * G * ((BD - E) - G)
            cout << " R3: A * T2 done" << endl;
            multiplyMatrices(T0, T1, T2, size);     // T4 = (A - E) * G * ((BD - E) - G) * BD
            cout << " R3: T0 * T1 done" << endl;
            MPI_Send(&(T2[0][0]), size*size, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD);   // T4 = (A - E) * G * ((BD - E) - G) * BD -> 1ому процессу
            cout << " R3: T2 -> 1 done" << endl;
            cout << "rank 3 ends" << endl;
            break;
        }
    }
    finalize(size);
}

#pragma region time
double measureRunTime (const int & size) {
    int wRank, wSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
    MPI_Comm_size(MPI_COMM_WORLD, &wSize);
    auto startTime = high_resolution_clock::now();
    run(size, wRank, wSize);
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime).count();
    return duration / 1e6; // Convert to milliseconds
}
#pragma endregion

int main() {
    ofstream resultFile("results.txt");
    int sizes[5] = {64, 256, 1024, 4096, 8192};
    MPI_Init(nullptr, nullptr);
    for (const auto & size:sizes) {
        double time = measureRunTime(size);
        resultFile << "Matrix size is: " << size << endl << "Execution time is: " << time << " milliseconds" << endl;
    }
    MPI_Finalize();
    resultFile.close(); // Close the output file
    return 0;
}
