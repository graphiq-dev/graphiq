const urls = ["/circuit_data"];
let circuit_data = {}

Promise.all(urls.map(url => d3.json(url))).then(run);

function run(dataset) {
    circuit_data = dataset
};

function myfunc() {
    console.log("Hello")
}