#include <iostream>
#include <chrono>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>

#include "adaptive_clustering.h"

#define SOCK_PATH "/tmp/pointcloudsock"

volatile sig_atomic_t keep_running = 1;
volatile int client_sock, server_sock;

void handle_sigint(int sig)
{
  keep_running = 0;
  shutdown(client_sock, SHUT_RDWR);
  shutdown(server_sock, SHUT_RDWR);
}

int main(int argc, const char **argv)
{
  std::string sensor_model = "HDL-32E";
  if(argc > 1) sensor_model = (argv[1]);

  AdaptiveClustering ac(sensor_model);

  signal(SIGINT, handle_sigint);

  int len, rc;
  int bytes_rec = 0;
  int backlog = 10;
  struct sockaddr_un server_sockaddr, peer_sock;
  struct sockaddr_un client_sockaddr;
  memset(&server_sockaddr, 0, sizeof(struct sockaddr_un));
  memset(&client_sockaddr, 0, sizeof(struct sockaddr_un));

  server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_sock == -1){
    std::cout << "SOCKET ERROR = " << strerror(errno) << std::endl;
    exit(1);
  }

  server_sockaddr.sun_family = AF_UNIX;
  strcpy(server_sockaddr.sun_path, SOCK_PATH); 
  len = sizeof(server_sockaddr);
  unlink(SOCK_PATH);
  rc = bind(server_sock, (struct sockaddr *) &server_sockaddr, len);
  if (rc == -1){
    std::cout << "BIND ERROR = " << strerror(errno) << std::endl;
    close(server_sock);
    exit(1);
  }

  rc = listen(server_sock, backlog);
  if (rc == -1){ 
    std::cout << "LISTEN ERROR = " << strerror(errno) << std::endl;
    close(server_sock);
    exit(1);
  }

  client_sock = accept(server_sock, (struct sockaddr *) &client_sockaddr, (socklen_t*) &len);
  if (client_sock == -1){
    std::cout << "ACCEPT ERROR = " << strerror(errno) << std::endl;
    close(server_sock);
    exit(1);
  }


  char num_points_str[17];
  while(keep_running){
    bytes_rec = recv(client_sock, &num_points_str, 16, 0);
    num_points_str[16] = '\0';
    if (bytes_rec == -1 || bytes_rec != 16){
      std::cout << "1 - Unexpected return " << bytes_rec << std::endl;
      break;
    }

    // skip leading zeros
    char* num_points_ptr = num_points_str;
    while(*num_points_ptr == '0')
      ++num_points_ptr;
    num_points_str[16] = '\0';
    int num_points = std::stoi(num_points_ptr);
    float *point_cloud = new float[num_points * 4];
    bytes_rec = 0;
    while(keep_running && bytes_rec < num_points * 4 * sizeof(float)){
      int ret = recv(client_sock, point_cloud, num_points * 4 * sizeof(float), 0);
      if (ret == -1 ){
        std::cout << "2 - Unexpected return " << bytes_rec << std::endl;
        break;
      }
      bytes_rec += ret;
    }
    auto start = std::chrono::steady_clock::now();
    auto clusters = ac.cluster(point_cloud, num_points*4);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
      << " ms" << std::endl;
    delete [] point_cloud;

  }

  close(client_sock);
  close(server_sock);

  return 0;
}
