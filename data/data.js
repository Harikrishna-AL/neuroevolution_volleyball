
var Dataset = require('./dataset.js');
var fs = require('fs');
/*
0 - circle
1 - xor
2 - 2 gaussians
3 - spiral
*/
var data_type_map = {
  0: 'circle',
  1: 'xor',
  2: '2 gaussians',
  3: 'spiral'
};

for (var i = 0; i < 4; i++) {
    Dataset.generateRandomData(i);
    var train_input = Dataset.getTrainData().w;
    var test_input = Dataset.getTestData().w;
    // split arr of size 400 into 200 data points with 2 featrues each
    train_input = train_input.reduce(function(result, value, index, array) {
    if (index % 2 === 0)
        result.push(array.slice(index, index + 2));
    return result;
    }, []);

    test_input = test_input.reduce(function(result, value, index, array) {
    if (index % 2 === 0)
        result.push(array.slice(index, index + 2));
    return result;
    }, []);

    var train_target = Dataset.getTrainLabel().w;
    var test_target = Dataset.getTestLabel().w;

    var train_data = {
        train_input: train_input,
        train_target: train_target
    };

    var test_data = {
        test_input: test_input,
        test_target: test_target
    };

    fs.writeFileSync('train_data_' + data_type_map[i] + '.json' , JSON.stringify(train_data));
    fs.writeFileSync('test_data_' + data_type_map[i] + '.json' , JSON.stringify(test_data));

    console.log('Data type: ' + data_type_map[i] + ' has been saved to data_' + data_type_map[i] + '.json');
}




