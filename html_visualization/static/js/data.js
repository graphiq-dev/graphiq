const urls = ["/circuit_data"];

Promise.all(urls.map(url => d3.json(url))).then(run);

function run(dataset) {
    circuit_data = dataset
};

let visualization_info = {
    register: {
        qreg: {p0: 50, p1: 100, p2: 150, p3: 200, e0: 250},
        creg: {c4: 300}
    },
    gates: [
        {
            gate_name: "H",
            qarg: "e0",
            control: "",
            params: {},
            x_pos: 50,
        },
        {
            gate_name: "My gate",
            qarg: "p1",
            control: "e0",
            params: {},
            x_pos: 100,
        },
    ]
}

console.log(visualization_info)