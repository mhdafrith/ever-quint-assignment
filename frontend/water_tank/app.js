function parseInput(s) {
    return s.split(",").map(x => parseInt(x.trim(), 10)).filter(x => !isNaN(x) && x >= 0);
}

function trapWater(arr) {
    // classic two-pass algorithm to compute water per index
    const n = arr.length;
    if (n === 0) return { total: 0, waterAt: [] };
    let leftMax = new Array(n), rightMax = new Array(n);
    leftMax[0] = arr[0];
    for (let i = 1; i < n; i++) leftMax[i] = Math.max(leftMax[i - 1], arr[i]);
    rightMax[n - 1] = arr[n - 1];
    for (let i = n - 2; i >= 0; i--) rightMax[i] = Math.max(rightMax[i + 1], arr[i]);
    let waterAt = new Array(n).fill(0);
    let total = 0;
    for (let i = 0; i < n; i++) {
        const w = Math.max(0, Math.min(leftMax[i], rightMax[i]) - arr[i]);
        waterAt[i] = w;
        total += w;
    }
    return { total, waterAt };
}

function createSVG(arr, waterAt, showWalls) {
    // Grid config
    const cellSize = 40;
    const maxHeight = Math.max(...arr, ...arr.map((h, i) => h + waterAt[i]), 0);
    const rows = maxHeight + 2; // +2 for detailed visual headroom
    const cols = arr.length;

    const width = cols * cellSize;
    const height = rows * cellSize;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", height);
    svg.style.maxWidth = `${width}px`;

    // Draw cells
    // Row index r: 0 is bottom.
    for (let r = rows - 1; r >= 0; r--) {
        for (let c = 0; c < cols; c++) {
            const x = c * cellSize;
            const y = (rows - 1 - r) * cellSize; // Invert y for SVG drawing (0 at top)

            const rect = document.createElementNS(svg.namespaceURI, "rect");
            rect.setAttribute("x", x);
            rect.setAttribute("y", y);
            rect.setAttribute("width", cellSize);
            rect.setAttribute("height", cellSize);
            rect.setAttribute("stroke", "#333"); // Grid lines
            rect.setAttribute("stroke-width", "1");

            const wallH = arr[c];
            const waterH = waterAt[c];

            if (r < wallH) {
                // Wall
                rect.setAttribute("fill", showWalls ? "#FFFF00" : "#FFFFFF"); // Yellow if walls shown, else White
            } else if (r < wallH + waterH) {
                // Water
                rect.setAttribute("fill", "#00B0F0"); // Bright Blue
            } else {
                // Empty
                rect.setAttribute("fill", "#FFFFFF"); // White
            }

            svg.appendChild(rect);
        }
    }
    return svg;
}

function draw(arr, waterAt, total) {
    const inputContainer = document.getElementById('chartInput');
    const outputContainer = document.getElementById('chartOutput');

    inputContainer.innerHTML = '';
    outputContainer.innerHTML = '';

    // Draw Input (Walls + Water)
    inputContainer.appendChild(createSVG(arr, waterAt, true));

    // Draw Output (Only Water)
    outputContainer.appendChild(createSVG(arr, waterAt, false));
}

function renderTable(arr, waterAt) {
    const container = document.getElementById('tableContainer');
    container.innerHTML = '';

    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    table.style.marginTop = '20px';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Index', 'Height', 'Water', 'Total Level'].forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    for (let i = 0; i < arr.length; i++) {
        const tr = document.createElement('tr');

        // Index
        const tdIndex = document.createElement('td');
        tdIndex.textContent = i;
        tdIndex.style.border = '1px solid #ddd';
        tdIndex.style.padding = '8px';
        tr.appendChild(tdIndex);

        // Height
        const tdHeight = document.createElement('td');
        tdHeight.textContent = arr[i];
        tdHeight.style.border = '1px solid #ddd';
        tdHeight.style.padding = '8px';
        tr.appendChild(tdHeight);

        // Water
        const tdWater = document.createElement('td');
        tdWater.textContent = waterAt[i];
        tdWater.style.border = '1px solid #ddd';
        tdWater.style.padding = '8px';
        if (waterAt[i] > 0) tdWater.style.backgroundColor = '#e1f5fe';
        tr.appendChild(tdWater);

        // Total Level
        const tdTotal = document.createElement('td');
        tdTotal.textContent = arr[i] + waterAt[i];
        tdTotal.style.border = '1px solid #ddd';
        tdTotal.style.padding = '8px';
        tr.appendChild(tdTotal);

        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
}

document.getElementById('runBtn').addEventListener('click', () => {
    const s = document.getElementById('arrInput').value;
    const arr = parseInput(s);
    const res = trapWater(arr);

    // Display Output clearly
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
    <div style="padding: 15px; background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 4px; margin-bottom: 20px;">
      <h3 style="margin: 0; color: #2e7d32;">Total Water Trapped: ${res.total} Units</h3>
    </div>
  `;

    draw(arr, res.waterAt, res.total);
    renderTable(arr, res.waterAt);

    document.getElementById('explain').textContent = `Input Heights: [${arr.join(', ')}]`;
});
