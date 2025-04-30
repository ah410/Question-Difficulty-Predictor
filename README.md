# Question Difficulty Predictor

A machine learning project to predict the difficulty of questions based on various features.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)

## Project Overview

The purpose of this project is for integration into my group's Capstone project. It is a learning enhancement system meant to predict how ready a student is to move on to harder questions based on current statistics gathered from their usage of the application.

## Features

- Predicts question difficulty
- Supports multiple input formats
- Easy integration with other tools

## Installation

```bash
git clone https://github.com/yourusername/question_difficulty_predictor.git
cd question_difficulty_predictor
```

## Usage

Run the following command to run the project which will go ahead and train as well as predict scores from some premade test data in `Main.java`.

```bash
mvn clean install exec:java
```

## Dataset

Generated dataset with random values between certain ranges. Example data below.

| timeSpent | correctLast3 | avgTime | confidence | quizScore | sessions | readinessScore | numberQuestionsAnswered |
|-----------|--------------|---------|------------|-----------|----------|----------------|--------------------------|
| 243.49    | 2            | 71.91   | 0.83       | 0.29      | 17       | 0.5563         | 6                        |
| 571.91    | 3            | 67.92   | 0.76       | 0.8       | 28       | 0.792          | 12                       |
| 447.24    | 1            | 22.56   | 0.57       | 1         | 9        | 0.6331         | 10                       |
| 371.24    | 3            | 54.12   | 0.96       | 0.03      | 2        | 0.4308         | 8                        |
| 118.93    | 3            | 58.02   | 0.2        | 0.9       | 25       | 0.7133         | 3                        |

