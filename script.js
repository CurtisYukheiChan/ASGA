document.getElementById('allocate_btn').onclick = async () => {
  try {
    const fileInput = document.getElementById('participant_file');
    const file = fileInput.files[0];
    if (!file) {
      alert('Please upload a CSV or XLSX file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_participants', document.getElementById('num_participants').value || '1');
    formData.append('group_size', document.getElementById('group_size').value || '1');
    formData.append('motivation_weight', document.getElementById('motivation_weight').value || '1');
    formData.append('academic_weight', document.getElementById('academic_weight').value || '0.5');
    formData.append('normalise_skill', document.getElementById('normalise_skill').checked);
    formData.append('ESL_weight', document.getElementById('ESL_weight').value || '0.8');
    formData.append('skill_weight', document.getElementById('skill_weight').value || '0.9');
    formData.append('min_skill', document.getElementById('min_skill').value || '0.8');
    formData.append('gender_weight', document.getElementById('gender_weight').value || '0.8');
    formData.append('blacklist_weight', document.getElementById('blacklist_weight').value || '2');
    formData.append('whitelist_weight', document.getElementById('whitelist_weight').value || '1');
    formData.append('blacklist_limit', document.getElementById('blacklist_limit').value || '2');
    formData.append('whitelist_limit', document.getElementById('gender_weight').value || '2');
    formData.append('teamwork_weight',document.getElementById('teamwork_weight').value || '2');
    // Solver settings
    formData.append('algorithm', document.getElementById('algorithm').value);
    formData.append('max_iterations', document.getElementById('max_iterations').value);
    formData.append('max_generations', document.getElementById('max_generations').value);
    formData.append('population_size', document.getElementById('population_size').value);
    // Output Settings
    formData.append('show_names', document.getElementById('show_names').checked);
    formData.append('show_email', document.getElementById('show_email').checked);
    formData.append('show_gender', document.getElementById('show_gender').checked);
    formData.append('show_ESL', document.getElementById('show_ESL').checked);
    formData.append('show_report', document.getElementById('show_report').checked);

    // Send allocation request
    const res = await fetch('/run_allocation', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      // Try to parse error JSON if any
      const errorData = await res.json().catch(() => ({}));
      alert('Allocation error: ' + (errorData.error || res.statusText));
      return;
    }

    const result = await res.json();
    console.log('Allocation result:', result);
    alert('Allocation complete! Check console or page for results.');

    // If user wants a PDF report, download it
    if (document.getElementById('show_report').checked) {
      const pdfRes = await fetch('/download_report');
if (!pdfRes.ok) {
  alert('Error downloading PDF report.');
  return;
}
const pdfBlob = await pdfRes.blob();

const url = window.URL.createObjectURL(pdfBlob);
const a = document.createElement('a');
a.href = url;
a.download = 'groups_report.pdf';
document.body.appendChild(a);
a.click();
a.remove();
window.URL.revokeObjectURL(url);
    }

    // You can also update your page here with `result` data if needed

  } catch (error) {
    alert('Error running allocation: ' + error.message);
  }
};
