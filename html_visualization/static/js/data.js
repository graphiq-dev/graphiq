const urls = ["/circuit_data"];

Promise.all(urls.map(url => d3.json(url))).then(run);

function run(dataset) {
    circuit_data = dataset
};

let visualization_info = {
    registers: {
        qreg: {p0: 50, p1: 100, p2: 150, p3: 200, e0: 250},
        creg: {c4: 300}
    },
    gates: [
        {
            gate_name: "H",
            qargs: ["e0"],
            controls: [],
            params: {},
            x_pos: 50,
        },
        {
            gate_name: "CX",
            qargs: ["p1"],
            controls: ["e0"],
            params: {},
            x_pos: 100,
        },
        {
            gate_name: "U",
            qargs: ["p2"],
            controls: ["e0"],
            params: {
                theta: "pi/2",
                phi: "pi/4",
                lambda: "pi/8",
            },
            x_pos: 200,
        }
    ]
}

console.log(visualization_info)