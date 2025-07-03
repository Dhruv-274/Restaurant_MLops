pipeline {
    agent any
    environment {
        dockerhubCred = credentials('docker-cred')
        dockerImage = 'restaurant-rating-predictor:latest'
    }
    stages {
        stage('Git checkout') {
            steps {
                script {
                    git(
                        url: 'https://github.com/Dhruv-274/Restaurant_MLops.git',
                        branch: 'master'
                    )
                }
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
    }
}
