# label -> acceptable gold first-party tools (any hit counts). Labels whose
# capability is not among the 49 core tools (browser, computer_*, mis_data,
# current_datetime, workflow panel, sessions, config...) are excluded.
GOLD = {
 "coding": ["write_file", "edit_file", "bash"], "bug_fix": ["edit_file", "read_file", "bash", "ripgrep"],
 "maintenance": ["edit_file", "read_file", "ripgrep"], "ssh": ["ssh"], "screenshot": ["screenshot"],
 "search": ["web_search"], "live_data": ["web_search", "web_fetch"], "live_data_visual": ["web_search"],
 "document_delivery": ["send_file", "send_to_im", "im_message"], "document_generate": ["generate_pdf"],
 "document_read": ["read_document", "read_excel", "read_pptx", "read_file", "FileRead"], "document_open": ["open"],
 "attachment_delivery": ["send_file"], "database": ["database", "database_query"], "office": ["office"],
 "knowledge_write": ["knowledge_save_text", "knowledge_save_url", "knowledge_import_files", "knowledge_import_directory", "knowledge_import_package", "knowledge_import_share"],
 "file_read": ["read_file", "list_directory", "Glob", "ripgrep", "FileRead"], "file_write": ["write_file", "edit_file", "edit_lines"],
 "file_delete": ["bash"], "shell_command": ["bash"], "git_inspect": ["bash"], "git_mutate": ["bash"],
 "audio_record": ["record_audio"], "audio_transcribe": ["asr"], "audio_synthesize": ["tts"], "audio_deliver": ["tts_render", "send_to_im"],
 "web_fetch": ["web_fetch"], "knowledge_read": ["knowledge_search", "knowledge_context_pack"], "app_launch": ["open"],
 "file_download": ["download_file"], "schedule_manage": ["manage_schedule"], "schedule_dispatch": ["manage_schedule"],
 "memory_manage": ["memory"], "task_track": ["task"], "goal_manage": ["goal"], "delegate_task": ["delegate_task"],
 "knowledge_admin": ["knowledge_export"],
}
SENSITIVE_TOOLS = {"ssh", "screenshot", "record_audio", "send_file", "send_to_im", "im_message", "database", "database_query",
                   "knowledge_save_text", "knowledge_save_url", "knowledge_import_files", "knowledge_import_directory",
                   "knowledge_import_package", "knowledge_import_share", "delegate_task", "memory", "manage_schedule", "bash", "write_file", "edit_file", "edit_lines"}
