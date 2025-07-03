pipeline {
    agent any
    environment {
        dockerhubCred = credentials('docker-cred')
        dockerImage = 'dhruvrs/restaurant-rating-predictor:latest'
    }
    stages {
        stage('Prepare') {
            steps {
                // Increase Git HTTP post buffer size to handle large repos
                sh 'git config --global http.postBuffer 524288000'
                // Clean workspace before checkout
                cleanWs()
            }
        }
        stage('Git checkout') {
            steps {
                checkout([$class: 'GitSCM',
                    branches: [[name: 'refs/heads/master']],
                    userRemoteConfigs: [[url: 'https://github.com/Dhruv-274/Restaurant_MLops.git']],
                    extensions: [[$class: 'CloneOption', depth: 1, noTags: false, shallow: true]]
                ])
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build(dockerImage)
                }
            }
        }
        stage('Push to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-cred') {
                        docker.image(dockerImage).push('latest')
                    }
                }
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    sh 'kubectl apply -f k8s/deployment.yaml'
                    sh 'kubectl apply -f k8s/service.yaml'
                }
            }
        }
    }
}
