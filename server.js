const express = require('express');
const path = require('path');
const cors = require('cors');  // Step 1: Import the cors package
const app = express();
const { exec } = require('child_process');
const bodyParser = require('body-parser');
const fs = require('fs');

// Step 2: Use the cors middleware
app.use(cors());

// Middleware for parsing JSON and ur                 lencoded form data
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Endpoint to handle form submission
app.post('/submit-form', (req, res) => {
    const formData = req.body;
    console.log(formData);

    fs.writeFile('data.json', JSON.stringify(formData, null, 2), 'utf8', (err) => {
        if (err) {
            console.error(err);
            res.status(500).send('Error saving data');
            return;
        }

        exec('python similarity_matrix.py', (error, stdout, stderr) => {
            if (error) {
                console.error(`Error executing Python script: ${error.message}`);
                res.status(500).send('Error executing Python script');
                return;
            }
            if (stderr) {
                console.error(`Python script encountered an error: ${stderr}`);
                res.status(500).send('Python script encountered an error');
                return;
            }
            console.log(`Python script output: ${stdout}`);
            res.status(200).send('Form submitted successfully');
        });
    });
});

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

app.get('/get-csv-data/:csvId', (req, res) => {
    const csvId = req.params.csvId;
    const csvFiles = {
        'csv1': path.join(__dirname, 'output.csv'),
        'csv2': path.join(__dirname, 'output_2.csv'),
        'csv3': path.join(__dirname, 'output_3.csv'),
        'csv4': path.join(__dirname, 'output_4.csv')
    };

    const csvFilePath = csvFiles[csvId];
    if (!csvFilePath) {
        res.status(404).send('CSV file not found');
        return;
    }

    fs.readFile(csvFilePath, 'utf8', (err, data) => {
        if (err) {
            console.error(err);
            res.status(500).send('Error reading CSV file');
            return;
        }

        res.send(data); // Send CSV data as a response
    });
});

const PORT = process.env.PORT || 2000;

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on http://0.0.0.0:${PORT}`);
});
